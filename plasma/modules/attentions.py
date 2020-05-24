import torch
import torch.nn as nn
import torch.nn.functional as func

from .commons import GlobalAverage


class SEAttention(nn.Module):

    def __init__(self, in_channels, ratio=0.5, rank=2, spatial=False):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.axes = list(range(2, 2 + rank))
        self.spatial = spatial

        op = nn.Conv2d if rank == 2 else nn.Conv3d
        self.channel_attention = nn.Sequential(*[
            GlobalAverage(rank, keepdims=True),
            op(in_channels, int(in_channels * ratio), kernel_size=1, groups=1),
            nn.ReLU(inplace=True),
            op(int(in_channels * ratio), in_channels, kernel_size=1, groups=1),
            nn.Sigmoid()
        ])

        if spatial:
            self.spatial_attention = nn.Sequential(*[
                op(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            ])

    def forward(self, x):
        att = self.channel_attention(x) * x

        if self.spatial:
            spatial_att = self.spatial_attention(x) * x
            att = att + spatial_att

        return att


class CBAM(nn.Module):

    def __init__(self, in_channels, ratio=1 / 16):
        super().__init__()

        self.spatial_avg = nn.AdaptiveAvgPool2d(1)
        self.spatial_max = nn.AdaptiveMaxPool2d(1)

        self.channel_attention = nn.Sequential(*[
            nn.Conv2d(in_channels, int(in_channels * ratio), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels * ratio), in_channels, kernel_size=1)
        ])

        self.spatial_attention = nn.Sequential(*[
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        ])

    def forward(self, x):
        spatial_avg = self.spatial_avg(x)
        spatial_max = self.spatial_max(x)
        channel_attention = self.channel_attention(spatial_max) + self.channel_attention(spatial_avg)
        channel_attention = channel_attention.sigmoid()

        x = channel_attention * x
        spatial_attention = torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)
        spatial_attention = self.spatial_attention(spatial_attention)
        x = spatial_attention * x

        return x


class SAModule(nn.Module):

    def __init__(self, num_channels):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat_map):
        batch_size, num_channels, height, width = feat_map.size()

        conv1_proj = self.conv1(feat_map).view(batch_size, -1,
                                               width * height).permute(0, 2, 1)

        conv2_proj = self.conv2(feat_map).view(batch_size, -1, width * height)

        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = func.softmax(relation_map, dim=-1)

        conv3_proj = self.conv3(feat_map).view(batch_size, -1, width * height)

        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(batch_size, num_channels, height, width)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map
