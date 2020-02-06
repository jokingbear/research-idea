import torch
import torch.nn as nn

from plasma.modules import *


class BN_ReLU_Conv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channels * groups)
        self.act = nn.ReLU(inplace=True)
        self.con = nn.Conv2d(in_channels * groups, out_channels * groups, kernel, stride, padding, groups=groups,
                             bias=bias)


class BN_ReLU_Routing(nn.Sequential):

    def __init__(self, in_channels, out_channels, groups=32, iters=3):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channels * groups)
        self.act = nn.ReLU(inplace=True)
        self.con = DynamicRouting2d(in_channels, out_channels, groups, iters, bias=False)


class ResBlock(nn.Module):

    def __init__(self, in_channels, bottleneck, out_channels, groups=32, iters=3, downsample=False):
        super().__init__()

        self.skip = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True) if downsample else Identity()

        self.con = nn.Sequential(*[
            BN_ReLU_Conv(in_channels, bottleneck * groups, kernel=1, padding=0),
            ChannelAttention(bottleneck, groups=groups, rank=2),
            BN_ReLU_Conv(bottleneck, bottleneck, stride=2 if downsample else 1, groups=groups),
            BN_ReLU_Routing(bottleneck, out_channels, groups, iters)
        ])

    def forward(self, x):
        con = self.con(x)
        skip = self.skip(x)

        return torch.cat([con, skip], dim=1)


class DenseCap(nn.Module):

    def __init__(self):
        super().__init__()

        bottleneck = 8
        iters = 1
        f0 = 64
        routing_feature = 32

        self.con1 = nn.Sequential(*[
            nn.Conv2d(1, f0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(f0),
            nn.ReLU(inplace=True),
        ])  # f0 x 256 x 256

        self.con2 = nn.Sequential(*[
            ResBlock(f0, bottleneck, routing_feature, iters=iters, downsample=True),
            ResBlock(f0 + routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(f0 + 2 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(f0 + 3 * routing_feature, bottleneck, routing_feature, iters=iters),
        ])  # (f0 + 4 * routing_feature) x 128 x 128

        self.con3 = nn.Sequential(*[
            BN_ReLU_Conv(f0 + 4 * routing_feature, 2 * f0, kernel=1, padding=0),
            ResBlock(2 * f0, bottleneck, routing_feature, iters=iters, downsample=True),
            ResBlock(2 * f0 + routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(2 * f0 + 2 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(2 * f0 + 3 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(2 * f0 + 4 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(2 * f0 + 5 * routing_feature, bottleneck, routing_feature, iters=iters),
        ])  # (2 * f0 + 6 * routing_feature) x 64 x 64

        self.con4 = nn.Sequential(*[
            BN_ReLU_Conv(2 * f0 + 6 * routing_feature, 4 * f0, kernel=1, padding=0),
            ResBlock(4 * f0, bottleneck, routing_feature, iters=iters, downsample=True),
            ResBlock(4 * f0 + routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 2 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 3 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 4 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 5 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 6 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 7 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 8 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(4 * f0 + 9 * routing_feature, bottleneck, routing_feature, iters=iters),
        ])  # (4 * f0 + 10 * routing_feature) x 32 x 32

        self.con5 = nn.Sequential(*[
            BN_ReLU_Conv(4 * f0 + 10 * routing_feature, 8 * f0, kernel=1, padding=0),
            ResBlock(8 * f0, bottleneck, routing_feature, iters=iters, downsample=True),
            ResBlock(8 * f0 + routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 2 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 3 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 4 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 5 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 6 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 7 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 8 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 9 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 10 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(8 * f0 + 11 * routing_feature, bottleneck, routing_feature, iters=iters),
        ])  # (8 * f0 + 12 * routing_feature) x 16 x 16

        self.con6 = nn.Sequential(*[
            BN_ReLU_Conv(8 * f0 + 12 * routing_feature, 16 * f0, kernel=1, padding=0),
            ResBlock(16 * f0, bottleneck, routing_feature, iters=iters, downsample=True),
            ResBlock(16 * f0 + routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(16 * f0 + 2 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(16 * f0 + 3 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(16 * f0 + 4 * routing_feature, bottleneck, routing_feature, iters=iters),
            ResBlock(16 * f0 + 5 * routing_feature, bottleneck, routing_feature, iters=iters),
            BN_ReLU_Conv(16 * f0 + 6 * routing_feature, 16 * f0, kernel=1, padding=0)
        ])  # (16 * f0) x 8 x 8

    def forward(self, x):
        con1 = self.con1(x)
        con2 = self.con2(con1)
        con3 = self.con3(con2)
        con4 = self.con4(con3)
        con5 = self.con5(con4)
        con6 = self.con6(con5)

        return con6


class Decoder(nn.Sequential):

    def __init__(self):
        super().__init__()

        f0 = 64

        self.con1 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BN_ReLU_Conv(16 * f0, 16 * f0),
            BN_ReLU_Conv(16 * f0, 16 * f0),
        ])  # 256 x 16 x 16

        self.con2 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BN_ReLU_Conv(16 * f0, 8 * f0),
            BN_ReLU_Conv(8 * f0, 8 * f0),
        ])  # 128 x 32 x 32

        self.con3 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BN_ReLU_Conv(8 * f0, 4 * f0),
            BN_ReLU_Conv(4 * f0, 4 * f0),
        ])  # 64 x 64 x 64

        self.con4 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BN_ReLU_Conv(4 * f0, 2 * f0),
            BN_ReLU_Conv(2 * f0, 2 * f0),
        ])  # 32 x 128 x 128

        self.con5 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BN_ReLU_Conv(2 * f0, f0),
            BN_ReLU_Conv(f0, f0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BN_ReLU_Conv(f0, 1, kernel=1, padding=0, bias=True),
            nn.Tanh()
        ])  # 1 x 512 x 512
