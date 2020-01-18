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

    def __init__(self, in_channels, ratio=0.5, rank=2):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.axes = list(range(2, 2 + rank))

        self.attention = nn.Sequential(*[
            nn.Linear(in_channels, int(in_channels * ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels * ratio), in_channels),
            nn.Sigmoid()
        ])

    def forward(self, x):
        global_avg = x.mean(dim=self.axes)
        att = self.attention(global_avg).view(-1, self.in_channels, *([1] * len(self.axes)))

        return x * att

    def extra_repr(self):
        return f"in_channels={self.in_channels}, ratio={self.ratio}, axes={self.axes}"
