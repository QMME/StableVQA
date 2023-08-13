import torch.nn as nn
from .Stripformer import Stripformer

def get_generator():
    
    model_g = Stripformer()
    return nn.DataParallel(model_g)

def get_nets(model_config):
    return get_generator(model_config)
