import torch
import torch.nn as nn
import torch.nn.functional as func

from .commons import GlobalAverage


class SEAttention(nn.Module):

    def __init__(self, in_channels, ratio=0.5, rank=2):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.rank = rank

        self.channel_attention = nn.Sequential(*[
            GlobalAverage(rank),
            nn.Linear(in_channels, int(in_channels * ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels * ratio), in_channels),
            nn.Sigmoid()
        ])

    def forward(self, x):
        att = self.channel_attention(x)
        result = att.reshape(*att.shape, *[1] * self.rank) * x

        return result


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
