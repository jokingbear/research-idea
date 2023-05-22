import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as vision_ops

import plasma.modules as modules


class CNBlock(nn.Module):

    def __init__(self, channels, p, layer_scale=1e-6):
        super().__init__()

        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1, 1) * layer_scale)

        self.block = nn.Sequential(*[
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels),
            modules.LayerNorm(channels, dim=1),
            nn.Conv3d(channels, channels * 2, kernel_size=1),
            modules.LayerNorm(channels * 2, dim=1),
            nn.GELU(),
            nn.Conv3d(channels * 2, channels, kernel_size=1),
        ])

        self.stochastic_depth = vision_ops.StochasticDepth(p, mode='row')

    def forward(self, inputs):
        residuals = self.layer_scale * self.block(inputs)
        residuals = self.stochastic_depth(residuals)
        results = inputs + residuals
        return results


class ImagenetNorm(nn.Module):

    def __init__(self):
        super().__init__()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

        mean = mean[:, np.newaxis, np.newaxis, np.newaxis]
        std = std[:, np.newaxis, np.newaxis, np.newaxis]

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, inputs):
        inputs = inputs  / 255

        return (inputs - self.mean) / self.std


class PhysNext(nn.Sequential):

    def __init__(self, t, fps, s=2,start_channel=32, p=0.1):
        super().__init__()

        self.normalization = ImagenetNorm()

        self.stem = nn.Sequential(*[
            nn.Conv3d(3, start_channel, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            modules.LayerNorm(start_channel, dim=1),
            CNBlock(start_channel, 0 / 7 * p),
            CNBlock(start_channel, 1 / 7 * p),
        ])  # T x H / 2 x W / 2

        self.encoder1 = nn.Sequential(*[
            modules.LayerNorm(start_channel, dim=1),
            nn.Conv3d(start_channel, start_channel * 2, kernel_size=2, stride=2),
            CNBlock(start_channel * 2, 2 / 7 * p),
            CNBlock(start_channel * 2, 3 / 7 * p),
        ])  # T / 2 x H / 4 x W / 4

        self.encoder2 = nn.Sequential(*[
            modules.LayerNorm(start_channel * 2, dim=1),
            nn.Conv3d(start_channel * 2, start_channel * 2, kernel_size=2, stride=2),
            CNBlock(start_channel * 2, 4 / 7 * p),
            CNBlock(start_channel * 2, 5 / 7 * p),
        ])  # T / 4 x H / 8 x W / 8

        self.decoder1 = nn.Sequential(*[
            modules.LayerNorm(start_channel * 2, dim=1),
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(start_channel * 2, start_channel * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            CNBlock(start_channel * 2, 6 / 7 * p),
        ])  # T / 2 x H / 8 x W / 8

        nframe_original = int(t * fps)
        nframe_modified = nframe_original // 4 * 4
        pad = nframe_original - nframe_modified

        self.decoder2 = nn.Sequential(*[
            modules.LayerNorm(start_channel * 2, dim=1),
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.ReplicationPad3d((0, pad, 0, 0, 0, 0)),
            nn.Conv3d(start_channel * 2, start_channel, kernel_size=(3, 1, 1), padding=(2, 0, 0)),
            CNBlock(start_channel, 7 / 7 * p),
        ])  # T x H / 8 x W / 8

        self.end = nn.Sequential(*[
            modules.LayerNorm(start_channel, dim=1),
            modules.AdaptivePooling3D((None, s, s)),
            nn.Conv3d(start_channel, 1, kernel_size=1),
        ])

    def forward(self, inputs):
        st_block = super().forward(inputs) # B x C x T x S x S
        st_block = st_block.flatten(start_dim=3)  # B x C x T x SS
        avg_block = st_block.mean(dim=-1, keepdim=True)  # B x C x T x 1

        if not self.training:
            return avg_block[..., 0]

        return torch.cat([st_block, avg_block], dim=-1)[:, 0]  # B x T x SS+1