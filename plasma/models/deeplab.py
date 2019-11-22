import torch
import torch.nn as nn
import torch.nn.functional as func

from plasma.modules import blocks


class DeepLab3(nn.Module):

    def __init__(self, in_filters=1, start_filters=64, groups=32, d=4, iters=1):
        super().__init__()

        f = start_filters
        self.con1 = nn.Sequential(*[
            blocks.ConvBlock(in_filters, f, kernel_size=7, stride=2, padding=3),
            blocks.ConvBlock(f, f * 2, kernel_size=7, stride=2, padding=3),
        ])  # 2f x 128 x 128

        self.con2 = nn.Sequential(*[
            blocks.ResidualBlock(f * 2, d, groups, iters, down_sample=True),
            blocks.ConvBlock(f * 4, f * 4),
            blocks.ResidualBlock(f * 4, d, groups, iters),
            blocks.ResidualBlock(f * 4, d, groups, iters),
        ])  # 4f x 64 x 64

        self.con3 = nn.Sequential(*[
            blocks.ResidualBlock(f * 4, d, groups, iters, down_sample=True),
            blocks.ConvBlock(f * 8, f * 8),
            blocks.ResidualBlock(f * 8, 2 * d, groups, iters),
            blocks.ResidualBlock(f * 8, 2 * d, groups, iters),
            blocks.ResidualBlock(f * 8, 2 * d, groups, iters),
        ])  # 8f x 32 x 32

        self.scale0 = nn.Sequential(*[
            blocks.ConvBlock(f * 8, groups * 2 * d, kernel_size=1, padding=0),
            blocks.ConvBlock(2 * d, 2 * d, groups=groups),
            blocks.RoutingBlock(2 * d, f * 4, groups, iters)
        ])  # 4f x 32 x 32

        self.scale1 = nn.Sequential(*[
            blocks.ConvBlock(f * 8, groups * 2 * d, kernel_size=1, padding=0),
            blocks.ConvBlock(2 * d, 2 * d, groups=groups, dilation=6, padding=6),
            blocks.RoutingBlock(2 * d, f * 4, groups, iters)
        ])  # 4f x 32 x 32

        self.scale2 = nn.Sequential(*[
            blocks.ConvBlock(f * 8, groups * 2 * d, kernel_size=1, padding=0),
            blocks.ConvBlock(2 * d, 2 * d, groups=groups, dilation=12, padding=12),
            blocks.RoutingBlock(2 * d, f * 4, groups, iters)
        ])  # 4f x 32 x 32

        self.scale3 = nn.Sequential(*[
            blocks.ConvBlock(f * 8, groups * 2 * d, kernel_size=1, padding=0),
            blocks.ConvBlock(2 * d, 2 * d, groups=groups, dilation=18, padding=18),
            blocks.RoutingBlock(2 * d, f * 4, groups, iters)
        ])  # 4f x 32 x 32

        self.final_embed = blocks.ConvBlock(f * 16, f, kernel_size=1, padding=0)
        self.middle_embed = blocks.ConvBlock(f * 2, f, kernel_size=1, padding=0)

        self.refine = nn.Sequential(*[
            nn.Conv2d(2 * f, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        con1 = self.con1(x)
        con2 = self.con2(con1)
        con3 = self.con3(con2)

        scale = torch.cat([
            self.scale0(con3),
            self.scale1(con3),
            self.scale2(con3),
            self.scale3(con3)
        ], dim=1)

        embed1 = self.final_embed(scale)
        embed1 = func.interpolate(embed1, scale_factor=4, mode="bilinear", align_corners=True)
        embed2 = self.middle_embed(con1)

        embed = torch.cat([embed1, embed2], dim=1)
        refine = self.refine(embed)
        refine = func.interpolate(refine, scale_factor=4, mode="bilinear", align_corners=True)

        return refine
