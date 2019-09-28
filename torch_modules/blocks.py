import torch.nn as nn

from torch_modules.router import DynamicRouting
from torch_modules.commons import MergeModule

conv_layer = nn.Conv2d
decon_layer = nn.ConvTranspose2d
router_layer = DynamicRouting
pooling_layer = nn.AvgPool2d


def normalize_deconvolution(x):
    return x[..., 1:, 1:]


def default_normalization(f):
    return nn.BatchNorm2d(f)


class Block(nn.Module):

    def __init__(self, out_filters, transformer, normalizer, activator):
        super().__init__()

        self.transfomer = transformer
        self.normalizer = normalizer or default_normalization(out_filters)
        self.activator = activator or (lambda arg: arg)

    def forward(self, *xs):
        transform = self.transfomer(xs[0])
        normalize = self.normalizer(transform)
        activate = self.activator(normalize)

        return activate


class ConvBlock(Block):

    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                 groups=1, dilation=1,
                 normalizer=None, activator=nn.LeakyReLU(0.2)):
        con = conv_layer(in_filters * groups, out_filters * groups, kernel_size, stride, padding,
                         dilation, groups, bias=False)

        super().__init__(out_filters, con, normalizer, activator)


class DeconBlock(Block):

    def __init__(self, in_filters, out_filters, kernel_size=3, stride=2, padding=0,
                 normalizer=None, activator=nn.LeakyReLU(0.2), has_shortcut=False):
        decon = decon_layer(in_filters, out_filters, kernel_size, stride, padding)

        super().__init__(out_filters, decon, normalizer, activator)

        self.merge = MergeModule() if has_shortcut else (lambda *args: args[0])

    def forward(self, x, *shortcuts):
        decon = normalize_deconvolution(super().forward(x))
        merge = self.merge(decon, *shortcuts)

        return merge


class RoutingBlock(Block):
    def __init__(self, in_filters, out_filters, groups, iters=3,
                 normalizer=None, activator=nn.LeakyReLU(0.2)):
        con = DynamicRouting(in_filters, out_filters, groups, iters, bias=False)

        super().__init__(out_filters, con, normalizer, activator)


class ResidualBlock(nn.Module):

    def __init__(self, in_filters, bottleneck, groups, iters=3, down_sample=False,
                 normalizers=(None,) * 3, activators=(nn.LeakyReLU(0.2),) * 3):
        super().__init__()

        self.res = nn.Sequential(*[
            ConvBlock(in_filters, bottleneck * groups, kernel_size=1, padding=0,
                      normalizer=normalizers[0], activator=activators[0]),
            ConvBlock(bottleneck, bottleneck, stride=2 if down_sample else 1, groups=groups,
                      normalizer=normalizers[1], activator=activators[1]),
            RoutingBlock(bottleneck, in_filters, groups, iters,
                         normalizer=normalizers[-1], activator=activators[-1] if down_sample else None)
        ])

        self.skipper = pooling_layer(kernel_size=2, stride=2) if down_sample else (lambda arg: arg)
        self.merger = MergeModule("concat" if down_sample else "add")
        self.activator = activators[-1] if not down_sample else (lambda arg: arg)

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
