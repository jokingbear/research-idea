import torch
import torch.nn as nn

from torch_modules.routing_module import DynamicRouting


rank = 2
conv_layer = nn.Conv2d
convt_layer = nn.ConvTranspose2d
router_layer = DynamicRouting
pooling_layer = nn.AvgPool2d


def normalize_deconvolution(x):
    return x[..., 1:, 1:]


def get_activation_layer(is_activated):
    return nn.LeakyReLU(0.2) if is_activated else (lambda x: x)


def get_normalization_layer(normalization):
    return normalization if normalization else (lambda x: x)


class ConvModule(nn.Module):

    def __init__(self, fi, fo, n_group=1, kernel=3, strides=1, padding=0, dilations=1,
                 activation=True, normalization=None):
        super().__init__()

        self.con = conv_layer(fi * n_group, fo * n_group, kernel,
                              stride=strides, padding=padding,
                              groups=n_group, dilation=dilations, bias=normalization is None)

        self.normalization = get_normalization_layer(normalization)
        self.activation = get_activation_layer(activation)

    def forward(self, x):
        y = self.con(x)
        y = self.normalization(y)
        y = self.activation(y)

        return y


class DeConvModule(nn.Module):

    def __init__(self, fi, fo, kernel=3, strides=2, activation=True, normalization=None):
        super().__init__()

        self.con = convt_layer(fi, fo, kernel, stride=strides, bias=normalization is None)

        self.normalization = get_normalization_layer(normalization)
        self.activation = get_activation_layer(activation)

    def forward(self, x):
        y = self.con(x)
        y = normalize_deconvolution(y)
        y = self.normalization(y)
        y = self.activation(y)

        return y


class RoutingModule(nn.Module):

    def __init__(self, n_group, fi, fo, n_iter=3, activation=True, normalization=None):
        super().__init__()

        self.router = router_layer(fi, fo, n_group, n_iter=n_iter, use_bias=normalization is None)
        self.normalization = get_normalization_layer(normalization)
        self.activation = get_activation_layer(activation)

    def forward(self, x):
        route = self.router(x)
        route = self.normalization(route)
        route = self.activation(route)

        return route


class MergeModule(nn.Module):

    def __init__(self, mode="concat"):
        super().__init__()
        self.mode = mode

    def forward(self, *xs):
        if self.mode == "concat":
            return torch.cat(tuple(xs), dim=1)
        else:
            final = xs[0]

            for x in xs[1:]:
                final += x

            return final


class ResidualModule(nn.Module):

    def __init__(self, n_group, fi, bottleneck, n_iter=3, down_sample=False, normalizations=(None, None, None)):
        super().__init__()

        self.con1 = ConvModule(fi, bottleneck * n_group, kernel=1, normalization=normalizations[0])
        self.con2 = ConvModule(bottleneck, bottleneck, n_group=n_group,
                               strides=2 if down_sample else 1, padding=1, normalization=normalizations[1])
        self.con3 = RoutingModule(n_group, bottleneck, fi, n_iter=n_iter, activation=down_sample,
                                  normalization=normalizations[-1])

        self.res_transform = pooling_layer(kernel_size=2, stride=2) if down_sample else lambda x: x
        self.res_combine = MergeModule(mode="concat" if down_sample else "add")
        self.activation = get_activation_layer(not down_sample)

    def forward(self, x):
        con1 = self.con1(x)
        con2 = self.con2(con1)
        con3 = self.con3(con2)

        x = self.res_transform(x)
        res = torch.cat((x, con3), dim=1) if self.down_sample else x + con3
        res = self.activation(res)

        return res
