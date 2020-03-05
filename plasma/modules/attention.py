import torch.nn as nn

from plasma.modules.commons import GlobalAverage


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
