import torch
import torch.nn as nn

from torch_modules import block_modules as blocks


def get_normalization_layer(f, n_group=1):
    return nn.InstanceNorm2d(f * n_group)


class DownModule(nn.Module):

    def __init__(self, fi, bottleneck, n_group=32, n_iter=3):
        super().__init__()

        self.con1 = blocks.ResidualModule(n_group, fi, bottleneck, n_iter=n_iter, down_sample=True,
                                          normalizations=[get_normalization_layer(bottleneck, n_group),
                                                          get_normalization_layer(bottleneck, n_group),
                                                          get_normalization_layer(fi)])
        self.con2 = blocks.ConvModule(2 * fi, 2 * fi, padding=1, normalization=get_normalization_layer(2 * fi))
        self.con3 = blocks.ResidualModule(n_group, 2*fi, bottleneck, n_iter=n_iter,
                                          normalizations=[get_normalization_layer(bottleneck, n_group),
                                                          get_normalization_layer(bottleneck, n_group),
                                                          get_normalization_layer(2*fi)])

    def forward(self, x):
        con1 = self.con1(x)
        con2 = self.con2(con1)
        con3 = self.con3(con2)

        return con3


class UpModule(nn.Module):

    def __init__(self, fi, bottleneck, n_group=32, n_iter=3):
        super().__init__()

        self.cont1 = blocks.DeConvModule(fi, fi // 2, normalization=get_normalization_layer(fi // 2))
        self.cont2 = blocks.ConvModule(fi, fi // 2, padding=1, normalization=get_normalization_layer(fi // 2))
        self.cont3 = blocks.ResidualModule(n_group, fi//2, bottleneck, n_iter=n_iter,
                                           normalizations=[get_normalization_layer(bottleneck, n_group),
                                                           get_normalization_layer(bottleneck, n_group),
                                                           get_normalization_layer(fi//2)])
        self.merge = blocks.MergeModule()

    def forward(self, x, shortcut):
        cont1 = self.cont1(x)
        concat = self.merge(cont1, shortcut)
        cont2 = self.cont2(concat)
        cont3 = self.cont3(cont2)

        return cont3


