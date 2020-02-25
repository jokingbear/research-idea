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


class ImagenetNormalization(nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std
