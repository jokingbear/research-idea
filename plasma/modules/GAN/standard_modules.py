import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class ScaleConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        coefficient = np.sqrt(2) / np.sqrt(in_channels * kernel * kernel)
        self.coefficient = torch.tensor(coefficient, dtype=torch.float)
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel, kernel), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None

        self.weight.normal_()

    def forward(self, x):
        coefficient = self.coefficient.to(x.device)

        return torch.conv2d(x, coefficient * self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel={self.kernel}, stride={self.stride}, padding={self.padding}, " \
               f"bias={self.bias is not None}"


class ScaleLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        coefficient = np.sqrt(2) / np.sqrt(in_features)
        self.coefficient = torch.tensor(coefficient, dtype=torch.float)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True) if bias else None

        self.weight.normal_()

    def forward(self, x):
        coefficient = self.coefficient.to(x.device)

        return func.linear(x, coefficient * self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class AdaNorm(nn.Module):

    def __init__(self, noise_features, out_features, normalization):
        super().__init__()

        self.normalization = normalization
        self.transformation = ScaleLinear(noise_features, 2 * out_features)

    def forward(self, x, z):
        gamma_beta = self.transformation(z).reshape([-1, 2, x.shape[1]])
        gamma = gamma_beta[:, 0, :]
        beta = gamma_beta[:, 1, :]

        return gamma * self.normalization(x) + beta


class Noise(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, in_features, 1, 1), requires_grad=True)

    def forward(self, x, noise=None):
        noise = noise or torch.randn(1, *x.shape[1:], device=x.device)

        return x + self.weight * noise


class ParallelGeneratorDiscriminator(nn.DataParallel):

    def forward(self, *inputs, **kwargs):
        d = {**kwargs}
        d["g_kwargs"]["batch_size"] = d["g_kwargs"]["batch_size"] // len(self.device_ids)

        return super().forward(*inputs, **d)


class GeneratorDiscriminator(nn.Module):

    def __init__(self, generator, discriminator):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, g_grad=True, g_kwargs=None, d_kwargs=None):
        g_kwargs = g_kwargs or {}
        d_kwargs = d_kwargs or {}

        if g_grad:
            imgs = self.generator(**g_kwargs)
        else:
            with torch.no_grad():
                imgs = self.generator(**g_kwargs)

        imgs = imgs if not torch.is_tensor(imgs) else [imgs]

        scores = self.discriminator(*imgs, **d_kwargs)

        return scores, imgs
