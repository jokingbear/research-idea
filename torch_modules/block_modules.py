import torch
import torch.nn as nn

from torch_modules.routing_module import DynamicRouting
from torch_modules.common_modules import MergeModule


rank = 2
conv_layer = nn.Conv2d
decon_layer = nn.ConvTranspose2d
router_layer = DynamicRouting
pooling_layer = nn.AvgPool2d


def normalize_deconvolution(x):
    return x[..., 1:, 1:]


def get_norm_layer(f, n_group=1):
    return nn.BatchNorm2d(f * n_group, eps=1E-7)


class BlockModule(nn.Module):

    def __init__(self, transform_layer, normalization=None, activation=nn.LeakyReLU(0.2)):
        super().__init__()

        normalization = normalization or get_norm_layer
        activation = activation or (lambda x: x)

        self.transform = transform_layer
        self.normalization = normalization
        self.activation = activation

    def forward(self, x):
        transform = self.transform(x)
        normalize = self.normalization(transform)
        activate = self.activation(normalize)

        return activate


class ConvModule(BlockModule):

    def __init__(self, fi, fo, kernel=3, stride=1, padding=1, n_group=1, dilation=1, **kwargs):
        super().__init__(conv_layer(fi * n_group, fo * n_group, kernel, stride=stride, padding=padding,
                                    groups=n_group, dilation=dilation, bias=False), **kwargs)


class DeConvModule(BlockModule):

    def __init__(self, fi, fo, kernel=3, stride=2, padding=0, has_shortcut=False, **kwargs):
        super().__init__(decon_layer(fi, fo, kernel, stride=stride, padding=padding, bias=False), **kwargs)

        self.merge = MergeModule() if has_shortcut else lambda *xs: xs[0]

    def forward(self, *xs):
        decon = super().forward(xs[0])
        merge = self.merge(*[decon, *xs[1:]])

        return merge


class RoutingModule(BlockModule):

    def __init__(self, n_group, fi, fo, n_iter=3, **kwargs):
        super().__init__(router_layer(fi, fo, n_group, n_iter=n_iter, use_bias=False), **kwargs)


class ResidualModule(nn.Module):

    def __init__(self, n_group, fi, bottleneck, n_iter=3, down_sample=False,
                 normalization=None, activation=nn.LeakyReLU(0.2)):
        super().__init__()

        self.res_path = nn.Sequential(
            ConvModule(fi, bottleneck * n_group, kernel=1, padding=0,
                       normalization=normalization, activation=activation),
            ConvModule(bottleneck, bottleneck, n_group=n_group, stride=2 if down_sample else 1,
                       normalization=normalization, activation=activation),
            RoutingModule(n_group, bottleneck, fi, n_iter=n_iter,
                          normalization=normalization, activation=activation if down_sample else None)
        )

        self.res_transform = pooling_layer(kernel_size=2, stride=2) if down_sample else lambda x: x
        self.res_combine = MergeModule(mode="concat" if down_sample else "add")
        self.activation = (lambda x: x) if down_sample else activation

    def forward(self, x):
        res_paths = self.res_path(x)
        res_trans = self.res_transform(x)

        merge = self.res_combine(res_paths, res_trans)
        activate = self.activation(merge)

        return activate
