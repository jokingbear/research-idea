import torch
import torch.nn as nn

from plasma.modules import *


class ResBlock(nn.Module):

    def __init__(self, in_channels, bottle_neck, out_channels):
        super().__init__()

        self.con = nn.Sequential(
            GroupMapping(in_channels, bottle_neck * 32),
            nn.ReLU(inplace=True),
            GroupConv2d(bottle_neck, bottle_neck, groups=32),
            nn.ReLU(inplace=True),
            GroupMapping(bottle_neck * 32, out_channels))
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        res = self.con(x)
        con = self.act(x + res)

        return con
