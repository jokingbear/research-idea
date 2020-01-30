import torch
import numpy as np
import torch.nn as nn

from plasma.modules import *


class ResBlock(nn.Module):

    def __init__(self, in_channels, bottle_neck, out_channels):
        super().__init__()

        self.con = nn.Sequential(
            GEMapping(in_channels, bottle_neck * 32),
            nn.ReLU(inplace=True),
            GEConv2d(bottle_neck, bottle_neck, groups=32),
            nn.ReLU(inplace=True),
            GEMapping(bottle_neck * 32, out_channels))

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.con(x)
        con = self.act(x + res)

        return con


net = nn.Sequential(*[
    PrimaryGroupConv2d(1, 8),
    nn.ReLU(inplace=True),
    ResBlock(8, 2, 8),
    GEConv2d(8, 16),
    nn.ReLU(inplace=True),
    Reshape(16, -1, 28, 28),
    GlobalAverage(rank=3)
])

net.cuda(0)

a = np.random.randn(28, 28)
b = np.rot90(a).copy()

with torch.no_grad():
    score_a = net(torch.tensor(a[np.newaxis, np.newaxis], dtype=torch.float, device="cuda:0"))
    score_b = net(torch.tensor(b[np.newaxis, np.newaxis], dtype=torch.float, device="cuda:0"))
    err = (score_a - score_b).abs().sum()
