import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.functional import adaptive_avg_pool3d
from functools import partial, reduce
from einops import rearrange
import numpy as np
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_tiny, swin_3d_small
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
from .xclip_backbone import build_x_clip_model
from .swin_backbone import SwinTransformer2D as ImageBackbone
from .swinv2_backbone import SwinTransformerV2
from .swinv1_backbone import SwinTransformer
from .head import VQAHead, IQAHead, VARHead
from .stripformer.networks import get_generator
from .resnet import generate_model
from .core.raft import RAFT

from .resnet2d import resnet50
import fastvqa.models.saverloader as saverloader

from PIL import Image
import cv2



class BaseEvaluator(nn.Module):
    def __init__(
        self,
        backbone=dict(),
        vqa_head=dict(),
    ):
        super().__init__()
        self.backbone = VideoBackbone(**backbone)
        self.vqa_head = VQAHead(**vqa_head)

    def forward(self, vclip, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(vclip)
                score = self.vqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns


class Stablev2Evaluator(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys = 'fragments,resize',
        multi=False,
        layer=-1,
        backbone=dict(),
        divide_head=False,
        vqa_head=dict(),
    ):
        super().__init__()

        self.multi = multi
        self.layer = layer
        self.blur = 8
        for key, hypers in backbone.items():
            print(backbone_size)
            
            if backbone_size=="divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == 'swin_tiny':
                b = swin_3d_tiny(in_chans=2, window_size=(4,4,4), **backbone[key])
            elif t_backbone_size == 'swin_tiny_grpb':
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == 'swin_tiny_grpb_m':
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4,4,4), frag_biases=[0,0,0,0])
            elif t_backbone_size == 'swin_small':
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == 'conv_tiny':
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == 'conv_small':
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == 'xclip':
                b = build_x_clip_model(**backbone[key])
            elif t_backbone_size == 'swinv2':
                b = SwinTransformerV2(img_size=224, window_size=7, num_classes=128)
            elif t_backbone_size == 'swinv1':
                b = SwinTransformer()
            elif t_backbone_size == 'resnet':
                b = resnet50(pretrained=True)
            else:
                raise NotImplementedError
            print("Setting backbone:", key+"_backbone")
            setattr(self, key+"_backbone", b) 
        
        self.deblur_net = get_generator()
        self.deblur_net.load_state_dict(torch.load('pretrained_weights/Stripformer_realblur_J.pth'))
        self.deblur_net = self.deblur_net.eval()

        self.motion_analyzer = generate_model(18, n_input_channels = 2, n_classes=256)

        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flow_model = RAFT()
        state_dict = torch.load('pretrained_weights/raft-things.pth')
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            t_key = key.replace("module.", "")
            i_state_dict[t_key] = state_dict[key]
        self.flow_model.load_state_dict(i_state_dict)
        self.flow_model.eval()

        
        self.quality = self.quality_regression(512 + 768 * 32 + 320 * self.blur, 128,1)

    def load_pretrained(self, state_dict):
        t_state_dict = self.resize_backbone.state_dict()
        for key, value in t_state_dict.items():
            if key in state_dict and state_dict[key].shape != value.shape:
                state_dict.pop(key)
        print(self.resize_backbone.load_state_dict(state_dict, strict=False))

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def get_blur_vec(self, model, frames, num):

        _, d, c, h, w = frames.shape

        with torch.no_grad():
            
            img_tensor = frames[:, 0:d:int(d/num), :, :, :]
            img_tensor = img_tensor.reshape(-1, c, h, w)

            factor = 8
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
            H, W = img_tensor.shape[2], img_tensor.shape[3]

            output = model(img_tensor)
        return output
    
    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():

                scores = []
                feats = {}

                for key in vclips:
                    n, c, d, h, w = vclips[key].shape
                    tmp = rearrange(vclips[key], "n c d h w -> n d c h w")
                    
                    x = vclips[key]
                    x = vclips[key].reshape(-1, c, h, w)
                    optical_flows = []
                
                    with torch.no_grad():
                       for i in range(d):
                           if (i+1 < d):                       
                               flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i+1, :, :])

                           else:
                               flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                           optical_flows.append(flow_up[0])


                    img_f = getattr(self, key.split("_")[0]+"_backbone")(x)
                    img_feat = img_f.reshape(n, d*img_f.size(1))

                    optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
                
                    blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                    total_feat = []
                    blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                    blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                    
                    total_feat.append(blur_feats)
                    total_feat.append(img_feat)
                    total_feat.append(optical_feat)

                    total_feat = torch.cat(total_feat, 1)
                    scores += [self.quality(total_feat)]
                    if return_pooled_feats:
                        feats[key] = img_feat
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x,y:x+y, scores)
                    else:
                        scores = scores[0]
            
                self.train()
                if return_pooled_feats:
                    return scores, feats
                return scores
                
        else:
            self.train()
            scores = []
            feats = {}

            for key in vclips:
                n, c, d, h, w = vclips[key].shape
                tmp = rearrange(vclips[key], "n c d h w -> n d c h w")
                
                x = vclips[key]
                x = vclips[key].reshape(-1, c, h, w)
                optical_flows = []
            
                with torch.no_grad():
                   for i in range(d):
                       if (i+1 < d):                       
                           flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i+1, :, :])
                           
                       else:
                           flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                       optical_flows.append(flow_up[0])
                    

                img_f = getattr(self, key.split("_")[0]+"_backbone")(x)
                img_feat = img_f.reshape(n, d*img_f.size(1))

                optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
            
                blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                total_feat = []
                blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                #
                total_feat.append(blur_feats)
                total_feat.append(img_feat)
                total_feat.append(optical_feat)

                total_feat = torch.cat(total_feat, 1)
                scores += [self.quality(total_feat)]
                if return_pooled_feats:
                    feats[key] = img_feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x,y:x+y, scores)
                else:
                    scores = scores[0]
            
            if return_pooled_feats:
                return scores, feats
            return scores


    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns

