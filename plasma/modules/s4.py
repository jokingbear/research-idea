import torch
import torch.nn as nn
import torch.nn.functional as func

import numpy as np
import pandas as pd

from ..resources import mapping as path_mapping

elements = torch.tensor(np.load(path_mapping.get('groups/groups.npy')), dtype=torch.float)

mapping = pd.read_csv(path_mapping.get('groups/cayley.csv'), index_col=0)

pairs = pd.read_json(path_mapping.get('groups/pairs.json'), typ="series")


class S4Prime(nn.Module):

    def __init__(self, in_channels, out_channels, group_grids, padding):
        super().__init__()

        assert isinstance(group_grids, nn.Parameter), 'group_grids needs to be parameter'
        assert not group_grids.requires_grad, 'group_grids must not require grad'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_channels = group_grids.shape[0]
        self.kernel_size = group_grids.shape[1]
        self.padding = padding

        self.grid = group_grids
        self._reset_parameters()

    def forward(self, vol):
        # x: B x in x D x H x W
        # flatten: out_in x k3
        #          G x out_in x k3
        flatten = torch.flatten(self.weight, end_dim=1)
        flatten = flatten[np.newaxis].expand(self.group_channels, -1, -1, -1, -1)

        # grid: G x 3 x 4
        grid = self.grid[pairs.values]

        # map_weight: G x out_in x k3
        #             G x out x in x k3
        #             out x G x in x k3
        #             out_G x in x k3
        map_weight = func.grid_sample(flatten, grid, align_corners=True)
        map_weight = map_weight.reshape(elements.shape[0], *self.weight.shape)
        map_weight = map_weight.transpose(0, 1)
        map_weight = map_weight.reshape(-1, *self.weight.shape[1:])

        # conv: B x outG x D x H x W
        #       B x out x G x D x H x W
        conv = func.conv3d(vol, map_weight, None, 1, self.padding)
        conv = conv.view(-1, self.out_channels, elements.shape[0], *conv.shape[-3:])
        return conv

    def _reset_parameters(self, nonlinearity='leaky_relu', a=0):
        k = self.kernel_size
        weight = torch.randn(self.out_channels, self.in_channels, k, k, k, dtype=torch.float)
        nn.init.kaiming_normal_(weight, nonlinearity=nonlinearity, a=a)

        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, group={self.group_channels}, " \
               f"kernel_size={self.kernel_size}, padding={self.padding}"


class S4Pool(nn.Module):

    def __init__(self, kind='average', kernel_size=2, stride=2):
        super().__init__()

        if kind == 'max':
            self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)
        else:
            self.pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        new_x = x.flatten(start_dim=1, end_dim=2)
        pooled = self.pool(new_x)
        return pooled.view(*x.shape[:3], *pooled.shape[-3:])


class S4Conv(nn.Module):

    def __init__(self, in_channels, out_channels, group_grids, padding, partition=1):
        super().__init__()

        assert isinstance(group_grids, nn.Parameter), 'group_grids needs to be parameter'
        assert not group_grids.requires_grad, 'group_grids must not require grad'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_channels = group_grids.shape[0]
        self.kernel_size = group_grids.shape[1]
        self.padding = padding
        self.partition = partition

        self._reset_parameters()
        self.grids = group_grids

    def forward(self, feature_maps):
        inverse_group = mapping.iloc[pairs.values]
        permute_weights = [self.weight[:, :, inverse_group.iloc[g].values] for g in range(self.group_channels)]

        # G x out x in x G x spatial
        new_weight = torch.stack(permute_weights, dim=0)
        new_weight = torch.flatten(new_weight, start_dim=1, end_dim=3)
        grid = self.grids[pairs.values]

        # G x out_in_G x spatial
        # G x out x in x G x spatial
        # out x G x in x G x spatial
        # out_G x in_G x spatial
        new_weight = func.grid_sample(new_weight, grid, align_corners=True)
        new_weight = new_weight.view(elements.shape[0], *self.weight.shape)
        new_weight = new_weight.transpose(0, 1)
        new_weight = new_weight.reshape(self.out_channels * elements.shape[0], -1, *self.weight.shape[-3:])

        # B x in_G x spatial
        feature_maps = feature_maps.flatten(start_dim=1, end_dim=2)

        # B x out_G x spatial
        # B x out x G x spatial
        conv = func.conv3d(feature_maps, new_weight, None, 1, self.padding, groups=self.partition)
        conv = conv.view(-1, self.out_channels, elements.shape[0], *conv.shape[-3:])
        return conv

    def _reset_parameters(self, nonlinearity='leaky_relu', a=0):
        k = self.kernel_size

        assert self.in_channels % self.partition == 0

        in_channels = self.in_channels // self.partition
        weight = torch.zeros(self.out_channels, elements.shape[0] * in_channels, *[k] * 3, dtype=torch.float)
        nn.init.kaiming_normal_(weight, nonlinearity=nonlinearity, a=a)
        weight = weight.reshape(self.out_channels, in_channels, elements.shape[0], *[k] * 3)

        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, group={self.group_channels}, " \
               f"kernel_size={self.kernel_size}, padding={self.padding}, partition={self.partition}"


class S4Linear(nn.Linear):

    def forward(self, x):
        # input: B x in x G x D x H x W
        transformed = torch.einsum('bigdhw,oi->bogdhw', x, self.weight)

        if self.bias is not None:
            bias = self.bias.view(1, -1, 1, 1, 1, 1)
            transformed = transformed + bias

        return transformed


class S4Norm(nn.Module):

    def __init__(self, channels, eps=1e-8):
        super().__init__()

        self.channels = channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(channels, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float), requires_grad=True)

    def forward(self, feature_maps):
        std, mean = torch.std_mean(feature_maps, dim=[-1, -2, -3, -4], keepdim=True)
        weight = self.weight.view(1, -1, 1, 1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1, 1, 1)

        return weight * (feature_maps - mean) / (std + self.eps) + bias

    def extra_repr(self):
        return f'channels={self.channels}, eps={self.eps}'


class S4Up(nn.Upsample):

    def forward(self, x):
        new_x = x.flatten(start_dim=1, end_dim=2)
        new_x = super().forward(new_x)
        new_x = new_x.view(*x.shape[:3], *new_x.shape[-3:])
        return new_x


def create_grid(kernel):
    half = kernel // 2
    coords = [-1 + i / half for i in range(half)] + [0] + [(i + 1) / half for i in range(half)]
    coords = np.array(coords)

    x_grid = coords[np.newaxis, np.newaxis]
    x_grid = np.tile(x_grid, [kernel, kernel, 1])

    y_grid = coords[np.newaxis, :, np.newaxis]
    y_grid = np.tile(y_grid, [kernel, 1, kernel])

    z_grid = coords[:, np.newaxis, np.newaxis]
    z_grid = np.tile(z_grid, [1, kernel, kernel])

    grids = np.stack([x_grid, y_grid, z_grid], axis=-1)
    grids = torch.tensor(grids, dtype=torch.float)

    affine_grids = torch.einsum("ijka,rba->rijkb", grids, elements)
    affine_grids = nn.Parameter(affine_grids, requires_grad=False)
    return affine_grids
