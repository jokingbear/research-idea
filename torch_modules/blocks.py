import torch.nn as nn

from torch_modules.router import DynamicRouting
from torch_modules.commons import MergeModule
from torch_modules import commons

rank = 2
conv_layer = nn.Conv2d
decon_layer = nn.ConvTranspose2d
router_layer = DynamicRouting
pooling_layer = nn.AvgPool2d
activation_layer = nn.LeakyReLU(0.2)


def normalize_deconvolution(x):
    return x[..., 1:, 1:]


def default_normalization(f):
    return nn.BatchNorm2d(f)


class Block(nn.Module):

    def __init__(self, out_filters, transformer, normalizer, has_activator):
        super().__init__()

        self.transformer = transformer
        self.normalizer = normalizer or default_normalization(out_filters)
        self.activator = activation_layer if has_activator else (lambda arg: arg)

    def forward(self, *xs):
        transform = self.transformer(xs[0])
        normalize = self.normalizer(transform)
        activate = self.activator(normalize)

        return activate


class DenseBlock(Block):

    def __init__(self, in_filters, out_filters, normalizer=None, dropout=None, has_activator=True):
        d = nn.Linear(in_filters, out_filters, bias=False)

        super().__init__(out_filters, d, normalizer, has_activator)

        self.dropout = nn.Dropout(p=dropout) if dropout else (lambda arg: arg)

    def forward(self, *xs):
        x = super().forward(*xs)

        return self.dropout(x)


class ConvBlock(Block):

    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                 groups=1, dilation=1,
                 normalizer=None, has_activator=True):
        con = conv_layer(in_filters * groups, out_filters * groups, kernel_size, stride, padding,
                         dilation, groups, bias=False)

        super().__init__(out_filters, con, normalizer, has_activator)


class DeconBlock(Block):

    def __init__(self, in_filters, out_filters, kernel_size=3, stride=2, padding=0,
                 normalizer=None, has_activator=True, has_shortcut=False):
        decon = decon_layer(in_filters, out_filters, kernel_size, stride, padding)

        super().__init__(out_filters, decon, normalizer, has_activator)

        self.merge = MergeModule() if has_shortcut else (lambda *args: args[0])

    def forward(self, x, *shortcuts):
        decon = normalize_deconvolution(super().forward(x))
        merge = self.merge(decon, *shortcuts)

        return merge


class RoutingBlock(Block):
    def __init__(self, in_filters, out_filters, groups, iters=3,
                 normalizer=None, has_activator=True):
        con = DynamicRouting(in_filters, out_filters, groups, iters, bias=False)

        super().__init__(out_filters, con, normalizer, has_activator)


class ResidualBlock(nn.Module):

    def __init__(self, in_filters, bottleneck, groups, iters=3, down_sample=False, normalizers=(None,) * 3):
        super().__init__()

        self.res = nn.Sequential(*[
            ConvBlock(in_filters, bottleneck * groups, kernel_size=1, padding=0, normalizer=normalizers[0]),
            ConvBlock(bottleneck, bottleneck, stride=2 if down_sample else 1, groups=groups, normalizer=normalizers[1]),
            RoutingBlock(bottleneck, in_filters, groups, iters, normalizer=normalizers[-1], has_activator=down_sample)
        ])

        self.skipper = pooling_layer(kernel_size=2, stride=2) if down_sample else (lambda arg: arg)
        self.merger = MergeModule("concat" if down_sample else "add")
        self.activator = activation_layer if not down_sample else (lambda arg: arg)

    def forward(self, x):
        res = self.res(x)
        skip = self.skipper(x)

        merge = self.merger(res, skip)
        activate = self.activator(merge)

        return activate


class MultiIOSequential(nn.Module):

    def __init__(self, *modules):
        super().__init__()

        [self.add_module(str(i), m) for i, m in enumerate(modules)]

    def forward(self, *inputs):
        for module in self.children():
            inputs = inputs if type(inputs) in {list, tuple} else [inputs]
            inputs = module(*inputs)

        return inputs


class SEBlock(nn.Module):

    def __init__(self, in_filters, reduction_rate=16):
        super().__init__()

        self.se = nn.Sequential(*[
            commons.GlobalAverage(rank, keepdims=True),
            conv_layer(in_filters, in_filters // reduction_rate, 1),
            activation_layer,
            conv_layer(in_filters // reduction_rate, in_filters, 1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        se = self.se(x)

        return x * se
