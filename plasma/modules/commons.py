import numpy as np
import torch
import torch.nn as nn


class GlobalAverage(nn.Module):

    def __init__(self, rank=2, keepdims=False):
        """
        :param rank: dimension of image
        :param keepdims: whether to preserve shape after averaging
        """
        super().__init__()
        self.axes = list(range(2, 2 + rank))
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, dim=self.axes, keepdim=self.keepdims)

    def extra_repr(self):
        return f"axes={self.axes}, keepdims={self.keepdims}"


class Reshape(nn.Module):

    def __init__(self, *shape):
        """
        final tensor forward result (B, *shape)
        :param shape: shape to resize to except batch dimension
        """
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

    def __init__(self, from_raw=True):
        """
        :param from_raw: whether the input image lies in the range of [0, 255]
        """
        super().__init__()

        self.from_raw = from_raw
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, x: torch.Tensor):
        if x.dtype != torch.float:
            x = x.float()

        x = x / 255 if self.from_raw else x
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)

        return (x - mean) / std

    def extra_repr(self):
        return f"from_raw={self.from_raw}"


class Normalization(nn.Module):

    def __init__(self, from_raw=True):
        """
        :param from_raw: whether the input image lies in the range of [0, 255]
        """
        super().__init__()

        self.from_raw = from_raw

    def forward(self, x):
        if x.dtype != torch.float:
            x = x.float()

        if self.from_raw:
            return x / 127.5 - 1
        else:
            return x * 2 - 1

    def extra_repr(self):
        return f"from_raw={self.from_raw}"


class ImageToTensor(nn.Module):

    def __init__(self, rank=2):
        super().__init__()

        self.rank = rank

    def forward(self, x):
        rank = len(x.shape[1:])

        if rank == self.rank:
            return x[:, np.newaxis]
        else:
            rank = self.rank
            return x.permute([0, rank + 1, *range(1, rank + 1)])

    def extra_repr(self):
        return f"rank={self.rank}"
