import torch
import torch.nn as nn
import torchvision.ops as vision_ops

import plasma.modules as modules


class CNBlock(nn.Module):

    def __init__(self, channels, p, layer_scale=1e-6):
        super().__init__()

        self.layer_scale = nn.Parameter(torch.ones(channels, 1) * layer_scale)

        self.block = nn.Sequential(*[
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels),
            modules.LayerNorm(channels, dim=1),
            nn.Conv1d(channels, channels * 4, kernel_size=1),
            modules.LayerNorm(channels * 4, dim=1),
            nn.GELU(),
            nn.Conv1d(channels * 4, channels, kernel_size=1),
        ])

        self.stochastic_depth = vision_ops.StochasticDepth(p, mode='row')

    def forward(self, inputs):
        residuals = self.layer_scale * self.block(inputs)
        residuals = self.stochastic_depth(residuals)
        results = inputs + residuals
        return results


class WaveNext(nn.Sequential):

    def __init__(self, in_channels=32, nblocks=(2, 3, 5), p=0.1):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv1d(1, in_channels, kernel_size=4, stride=4),
            modules.LayerNorm(in_channels, dim=1),
        ])

        blocks = []
        prev_channel = in_channels
        i = 0
        total = sum(nblocks) - 1
        for ridx, nblock in enumerate(nblocks):
            for _ in range(nblock):
                blocks.append(CNBlock(prev_channel, i / total * p))
                i += 1

            if ridx < len(nblocks) - 1:
                blocks.append(modules.LayerNorm(prev_channel, dim=1))
                blocks.append(nn.Conv1d(prev_channel, prev_channel * 2, kernel_size=2, stride=2))
                prev_channel *= 2

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(*[
            modules.GlobalAverage(dims=[-1]),
            nn.Linear(prev_channel, 2),
        ])
