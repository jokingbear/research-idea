import torch.nn as nn

from plasma.modules.commons import GlobalAverage


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=0.5, rank=2, groups=1):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.axes = list(range(2, 2 + rank))
        self.groups = groups

        op = nn.Conv2d if rank == 2 else nn.Conv3d

        self.attention = nn.Sequential(*[
            GlobalAverage(rank),
            op(in_channels * groups, groups * int(in_channels * ratio), kernel_size=1, groups=groups),
            nn.ReLU(inplace=True),
            op(groups * int(in_channels * ratio), groups * in_channels, kernel_size=1, groups=groups),
            nn.Sigmoid()
        ])

    def forward(self, x):
        att = self.attention(x)

        return x * att

    def extra_repr(self):
        return f"in_channels={self.in_channels}, ratio={self.ratio}, axes={self.axes}, groups={self.groups}"


class SpatialAttention(nn.Module):

    def __init__(self, in_channels, rank=2):
        super().__init__()

        op = nn.Conv2d if rank == 2 else nn.Conv3d
        
        self.att = nn.Sequential(*[
            op(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        att = self.att(x)

        return att * x
