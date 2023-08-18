# StableVQA

可复现论文["StableVQA: A Deep No-Reference Quality Assessment Model for Video Stability"](https://arxiv.org/abs/2308.04904)

## Database

抖动视频数据集下载[StableDB](https://drive.google.com/file/d/1XO1tkmSNg-yPcfQ0WSnpvB3mu0bILZQA/view?usp=drive_link)

## Pretrained Weights

下载RAFT预训练权重[raft-things.pth](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

下载StripFormer预训练权重[Stripformer_realblur_J.pth](https://drive.google.com/drive/folders/1YcIwqlgWQw_dhy_h0fqZlnKGptq1eVjZ?usp=sharing)

下载Swin Transformer预训练权重[swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

下载StableVQA预训练权重[swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

默认路径为`./pretrained_weights/*.pth`

## Training

    python new_train.py -o ./options/stable.yml

## Testing

    python new_test.py -o ./options/stable.yml
