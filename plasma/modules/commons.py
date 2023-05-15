import numpy as np
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

    def __init__(self, *shape, include_batch_dim=0):
        """
        final tensor forward result (B, *shape)
        :param shape: shape to resize to except batch dimension
        """
        super().__init__()

        self.shape = shape
        self.include_batch_dim = include_batch_dim

    def forward(self, x):
        shapes = self.shape if self.include_batch_dim else [x.shape[0], *self.shape]
        return x.reshape(shapes)

    def extra_repr(self):
        return f"shape={self.shape}, include_batch_dim={self.include_batch_dim}"


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

        mean = self.mean.view(1, -1, *[1] * len(x.shape[2:]))
        std = self.std.view(1, -1, *[1] * len(x.shape[2:]))
        return (x - mean) / std

    def extra_repr(self):
        return f"mean={self.mean}, std={self.std}"


class Permute(nn.Module):

    def __init__(self, *permutation):
        super().__init__()

        self.permutation = permutation
    
    def forward(self, inputs):
        return inputs.permute(self.permutation)

    def extra_repr(self) -> str:
        return f'permuration={self.permutation}'


class LayerNorm(nn.Module):

    def __init__(self, channels, dim=-1, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        ndim = len(x.shape)

        if self.dim < 0:
            dim = ndim + self.dim
        else:
            dim = self.dim

        weight = self.weight.view(*[1] * dim, -1, *[1] * (ndim - dim - 1))
        bias = self.bias.view(*[1] * dim, -1, *[1] * (ndim - dim - 1))

        std, mean = torch.std_mean(x, dim=self.dim, keepdim=True)
        return weight * (x - mean) / (std + self.eps) + bias

    def extra_repr(self) -> str:
        return f'channels={self.weight.shape[0]}, dim={self.dim}, eps={self.eps}'


class AdaptivePooling3D(nn.Module):

    def __init__(self, final_shape):
        super().__init__()

        self.final_shape = final_shape

    def forward(self, x):
        s1, s2, s3 = self.final_shape

        B, C, S1, S2, S3 = x.shape

        s1 = s1 or S1
        s2 = s2 or S2
        s3 = s3 or S3

        k1 = S1 // s1
        k2 = S2 // s2
        k3 = S3 // s3

        results = x.reshape(B, C, s1, k1, s2, k2, s3, k3)
        return results.mean(dim=[-1, -3, -5])

    def extra_repr(self) -> str:
        return f'final_shape={self.final_shape}'
