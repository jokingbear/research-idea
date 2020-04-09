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


class ImagenetNorm(nn.Module):

    def __init__(self, from_gray=True, from_raw=True):
        super().__init__()

        self.from_gray = from_gray
        self.from_raw = from_raw
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1) if self.from_gray else x
        x = x / 255 if self.from_raw else x
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)

        return (x - mean) / std

    def extra_repr(self):
        return f"from_gray={self.from_gray}, from_raw={self.from_raw}"

