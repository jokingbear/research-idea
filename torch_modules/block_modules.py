import torch
import torch.nn as nn
import torch_modules.visualize as viz

from torch_modules.routing_module import DynamicRouting
from torch_modules.common_modules import MergeModule


conv_layer = nn.Conv2d
decon_layer = nn.ConvTranspose2d
router_layer = DynamicRouting
pooling_layer = nn.AvgPool2d


def normalize_deconvolution(x):
    return x[..., 1:, 1:]


def get_norm_layer(f, n_group=1):
    return nn.BatchNorm2d(f * n_group, eps=1E-7)


class BlockModule(nn.Module, viz.GraphModule):

    def __init__(self, fo, n_group, transform_layer, normalization=None, activation=nn.LeakyReLU(0.2)):
        super().__init__()

        normalization = normalization or get_norm_layer
        activation = activation or (lambda x: x)

        self.transform = transform_layer
        self.normalization = normalization(fo, n_group)
        self.activation = activation

    def forward(self, x):
        transform = self.transform(x)
        normalize = self.normalization(transform)
        activate = self.activation(normalize)

        return activate

    def graph_forward(self, graph, input_nodes, input_tensors):
        transform, val = viz.render_module(self.transform, graph, input_tensors, input_nodes)
        normalize, val = viz.render_module(self.normalization, graph, val, transform)

        if isinstance(self.activation, nn.Module):
            return viz.render_module(self.activation, graph, val, normalize)

        return normalize, val


class ConvModule(BlockModule):

    def __init__(self, fi, fo, kernel=3, stride=1, padding=1, n_group=1, dilation=1, **kwargs):
        con = conv_layer(fi * n_group, fo * n_group, kernel, stride=stride, padding=padding,
                         groups=n_group, dilation=dilation, bias=False)
        super().__init__(fo, n_group, con, **kwargs)


class DeConvModule(BlockModule):

    def __init__(self, fi, fo, kernel=3, stride=2, padding=0, has_shortcut=False, **kwargs):
        decon = decon_layer(fi, fo, kernel, stride=stride, padding=padding, bias=False)
        super().__init__(fo, 1, decon, **kwargs)

        self.merge = MergeModule() if has_shortcut else lambda *xs: xs[0]

    def forward(self, *xs):
        x = xs[0]
        shortcut = xs[1:]

        decon = super().forward(x)
        decon = normalize_deconvolution(decon)
        merge = self.merge(*[decon, *shortcut])

        return merge

    def graph_forward(self, graph, input_nodes, input_tensors):
        decon, val = super().graph_forward(graph, input_nodes[0:1], input_tensors[0:1])
        val = [normalize_deconvolution(val[0])]

        if isinstance(self.merge, nn.Module):
            return viz.render_module(self.merge, graph, val + input_tensors[1:], decon + input_nodes[1:])
        else:
            return decon, val


class RoutingModule(BlockModule):

    def __init__(self, n_group, fi, fo, n_iter=3, **kwargs):
        router = router_layer(fi, fo, n_group, n_iter, bias=False)
        super().__init__(fo, 1, router, **kwargs)


class ResidualModule(nn.Module, viz.GraphModule):

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

    def graph_forward(self, graph, input_nodes, input_tensors):
        res_paths, res_val = viz.render_module(self.res_path, graph, input_tensors, input_nodes)

        if isinstance(self.res_transform, nn.Module):
            input_nodes, input_tensors = viz.render_module(self.res_transform, graph, input_tensors, input_nodes)

        merge, val = viz.render_module(self.res_combine, graph, input_tensors + res_val, input_nodes + res_paths)

        if isinstance(self.activation, nn.Module):
            return viz.render_module(self.activation, graph, val, merge)

        return merge, val


class MultiIOSequential(nn.Module, viz.GraphModule):

    def __init__(self, *modules):
        super().__init__()

        [self.add_module(str(idx), m) for idx, m in enumerate(modules)]

    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = inputs if type(inputs) in {tuple, list} else [inputs]
            inputs = module(*inputs)

        return inputs

    def graph_forward(self, graph, input_nodes, input_tensors):
        for module in self._modules.values():
            input_nodes, input_tensors = viz.render_module(module, graph, input_tensors, input_nodes)

        return input_nodes, input_tensors
