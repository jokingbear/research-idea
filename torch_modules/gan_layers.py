import torch
import torch.nn as nn
import torch.nn.functional as func

import numpy as np


rank = 2
con_op = func.conv2d


class ScaleCon(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.stride = stride
        self.padding = padding

        spatial_shape = [1] * rank
        kernel_shape = [kernel_size] * rank
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_shape))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, *spatial_shape)) if bias else None
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

        self.weight = nn.Parameter(torch.tensor(out_channels, in_channels))
        self.bias = nn.Parameter(torch.tensor(out_channels)) if bias else None
        self.const = np.sqrt(2 / in_channels)

        self.reset_parameters()

    def forward(self, x):
        return func.linear(x, self.const * self.weight, self.bias)

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def extra_repr(self):
        f_out, f_in = self.weight.shape
        return f"in_channels={f_in}, out_channels={f_out}, bias={self.bias is not None}"
