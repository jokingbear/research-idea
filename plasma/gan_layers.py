import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

rank = 2
con_op = func.conv2d


class ScaleCon(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.stride = stride
        self.padding = padding

        kernel_shape = [kernel_size] * rank
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, *kernel_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None
        self.const = np.sqrt(2 / (in_channels * kernel_size**rank))

        self.reset_parameters()

    def forward(self, x):
        return con_op(x, self.const * self.weight, self.bias, self.stride, self.padding)

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def extra_repr(self):
        f_out, f_in = self.weight.shape[:2]
        k = self.weight.shape[-1]
        return f"in_channels={f_in}, out_channels={f_out}, kernel_size={k}, " \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}"


class ScaleLinear(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None
        self.const = np.sqrt(2 / in_channels)

        self.reset_parameters()

    def forward(self, x):
        return func.linear(x, self.const * self.weight, self.bias)

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def extra_repr(self):
        f_out, f_in = self.weight.shape
        return f"in_channels={f_in}, out_channels={f_out}, bias={self.bias is not None}"


class MiniBatchStd(nn.Module):

    def forward(self, x):
        var = x.var(dim=0).mean()
        std = torch.sqrt(var + 1e-5).expand(x.shape[0], 1, *x.shape[2:])

        return torch.cat([x, std], dim=1)


class NoiseInjector(nn.Module):

    def __init__(self, in_filters):
        super().__init__()

        self.in_filters = in_filters
        self.weight = nn.Parameter(torch.zeros(1, in_filters, 1, 1), requires_grad=True)

    def forward(self, x, noise=None):
        noise = torch.randn(x.shape[0], 1, x.shape[1], x.shape[2], device=x.device)

        return x + self.weight * noise


class AdaN(nn.Module):

    def __init__(self, normalization):
        super().__init__()

        self.normalization = normalization

    def forward(self, x, ys, yb):
        return ys * self.normalization(x) + yb


class Const(nn.Module):

    def __init__(self, *shape, const=1):
        super().__init__()

        self.const = nn.Parameter(const * torch.ones(1, *shape), requires_grad=True)

    def forward(self, batch_size=1):
        return self.const.expand(batch_size, -1, -1, -1)
