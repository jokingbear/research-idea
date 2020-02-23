import torch
import torch.nn as nn


class GlobalAverage(nn.Module):

    def __init__(self, rank=2, keepdims=False):
        super().__init__()
        self.axes = list(range(2, 2 + rank))
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, dim=self.axes, keepdim=self.keepdims)

    def extra_repr(self):
        return f"axes={self.axes}, keepdims={self.keepdims}"


class Reshape(nn.Module):

    def __init__(self, *shape):
        super().__init__()

        self.shape = shape

    def forward(self, x):
        return x.reshape([x.shape[0], *self.shape])

    def extra_repr(self):
        return f"shape={self.shape}"


class Identity(nn.Module):

    def forward(self, x):
        return x


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=0.5, rank=2, groups=1):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.axes = list(range(2, 2 + rank))
        self.groups = groups

        op = nn.Conv2d if rank == 2 else nn.Conv3d

        self.attention = nn.Sequential(*[
            GlobalAverage(rank, keepdims=True),
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


class ImagenetNormalization(nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std
