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
        return inputs.permuate(self.permutation)

    def extra_repr(self) -> str:
        return f'permuration={self.permutation}'
