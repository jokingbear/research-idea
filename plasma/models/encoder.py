import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

from plasma import blocks


class ResCap(nn.Sequential):

    def __init__(self, corr_matrix, in_filters=1, start_filters=64, groups=32, d=4, iters=1):
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

        self.con4 = nn.Sequential(*[
            blocks.ResidualBlock(f * 8, 2 * d, groups, iters, down_sample=True),
            blocks.ConvBlock(f * 16, f * 16),
            blocks.ResidualBlock(f * 16, 4 * d, groups, iters),
            blocks.ResidualBlock(f * 16, 4 * d, groups, iters),
            blocks.ResidualBlock(f * 16, 4 * d, groups, iters),
            blocks.ResidualBlock(f * 16, 4 * d, groups, iters),
        ])  # 16f x 16 x 16

        self.con5 = nn.Sequential(*[
            blocks.ResidualBlock(f * 16, 4 * d, groups, iters, down_sample=True),
            blocks.ConvBlock(f * 32, f * 32),
            blocks.ResidualBlock(f * 32, 8 * d, groups, iters),
            blocks.ResidualBlock(f * 32, 8 * d, groups, iters),
            blocks.ResidualBlock(f * 32, 8 * d, groups, iters),
        ])  # 32f x 8 x 8

        self.features = blocks.commons.GlobalAverage()

        self.classifier = GraphClassifier(corr_matrix)


class GraphTransform(nn.Module):

    def __init__(self, corr_matrix, in_filters, out_filters):
        super().__init__()

        self.corr_matrix = torch.tensor(corr_matrix, dtype=torch.float)
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.weights = nn.Parameter(torch.zeros(in_filters, out_filters), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_filters), requires_grad=True)

        self.init_param()

    def forward(self, x):
        corr = self.corr_matrix.to(x.device)
        info_prop = torch.matmul(corr, x)
        affine = torch.matmul(info_prop, self.weights) + self.bias

        return affine

    def init_param(self):
        nn.init.kaiming_normal_(self.weights)


class GraphClassifier(nn.Module):

    def __init__(self, corr_matrix, n_class=14):
        super().__init__()

        one_hot = torch.tensor(np.array(range(14)), dtype=torch.long)
        one_hot = func.one_hot(one_hot, num_classes=n_class).float()
        self.one_hot = one_hot

        self.arch = nn.Sequential(*[
            GraphTransform(corr_matrix, n_class, 1024),
            nn.LeakyReLU(0.2),
            GraphTransform(corr_matrix, 1024, 1024),
            nn.LeakyReLU(0.2),
            GraphTransform(corr_matrix, 1024, 2048)
        ])

    def forward(self, x):
        embed = self.one_hot.to(x.device)
        embed = self.arch(embed)

        align = torch.matmul(x, embed.transpose(0, 1))
        return torch.sigmoid(align)
