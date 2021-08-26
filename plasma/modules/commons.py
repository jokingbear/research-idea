import torch
import torch.nn as nn


class GlobalAverage(nn.Module):

    def __init__(self, dims=(-1, -2), keepdims=False):
        """
        :param dims: dimensions of inputs
        :param keepdims: whether to preserve shape after averaging
        """
        super().__init__()
        self.dims = dims
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, dim=self.dims, keepdim=self.keepdims)

    def extra_repr(self):
        return f"dims={self.dims}, keepdims={self.keepdims}"


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

    def __init__(self, mean=127.5, std=127.5):
        """
        calculate (x - mean) / std
        :param mean: mean
        :param std: std
        """
        super().__init__()

        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std, dtype=torch.float), requires_grad=False)

    def forward(self, x):
        if x.dtype != torch.float:
            x = x.float()

        mean = self.mean.view(1, -1, *[1] * x.shape[2:])
        std = self.std.view(1, -1, *[1] * x.shape[2:])
        return (x - mean) / std

    def extra_repr(self):
        return f"mean={self.mean}, std={self.std}"


class ClipHU(nn.Module):

    def __init__(self, clip_min, clip_max):
        super().__init__()

        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, vol: torch.Tensor):
        return vol.clamp(self.clip_min, self.clip_max)

    def extra_repr(self):
        return f'clip_min={self.clip_min}, clip_max={self.clip_max}'


class LocalNorm(nn.Module):

    def __init__(self, dims=-1, eps=1e-8):
        super().__init__()

        self.dims = dims
        self.eps = eps

    def foward(self, x):
        std, mean = torch.std_mean(x, self.dims, keepdim=True)

        return (x - mean) / (std + self.eps)

    def extra_repr(self):
        return f'dims={self.dims}, eps={self.eps}'
