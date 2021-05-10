import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

from .commons import GlobalAverage


class GConvPrime(nn.Module):

    def __init__(self, in_channels, out_channels, group_grids, pairs, kernel_size=3, stride=1, padding=1):
        super().__init__()

        assert isinstance(group_grids, nn.Parameter), 'group_grids needs to be parameter'
        assert not group_grids.requires_grad, 'group_grids must not require grad'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_channels = group_grids.shape[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pairs = pairs

        self.grid = group_grids
        self._init_parameter()

    def forward(self, vol):
        # x: B x Cin x D x H x W
        # flatten: CoutCin x k3
        #          G x CoutCin x k3
        flatten = self.weight.flatten(end_dim=1)
        flatten = torch.stack([flatten] * self.group_channels, dim=0)

        # grid: G x 3 x 4
        grid = self.grid[self.pairs.values]

        # map_weight: G x CoutCin x k3
        #             GCout x Cin x k3
        map_weight = func.grid_sample(flatten, grid, align_corners=True)
        map_weight = map_weight.reshape(-1, *self.weight.shape[1:])

        # conv: B x GCout x D x H x W
        #       B x G x Cout x D x H x W
        conv = func.conv3d(vol, map_weight, None, self.stride, self.padding)
        conv = conv.reshape(-1, self.group_channels, self.out_channels, *conv.shape[2:])
        return conv

    def _init_parameter(self, nonlinearity='relu'):
        k = self.kernel_size
        weight = torch.randn(self.out_channels, self.in_channels, k, k, k, dtype=torch.float)
        nn.init.kaiming_normal_(weight, nonlinearity=nonlinearity)

        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, group={self.group_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class GConv(nn.Module):

    def __init__(self, in_channels, out_channels, group_grids, mapping, pairs, kernel_size=3, stride=1, padding=1):
        super().__init__()

        assert isinstance(group_grids, nn.Parameter), 'group_grids needs to be parameter'
        assert not group_grids.requires_grad, 'group_grids must not require grad'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_channels = group_grids.shape[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.mapping = mapping
        self.pairs = pairs

        self._create_weight()
        self.grids = group_grids

    def forward(self, feature_maps):
        inverse_group = self.mapping.iloc[self.pairs.values]
        permute_weights = [self.weight[:, inverse_group.iloc[g].values] for g in range(self.group_channels)]

        # G x out x G x in x spatial
        new_weight = torch.stack(permute_weights, dim=0)
        new_weight = torch.flatten(new_weight, start_dim=1, end_dim=3)
        grid = self.grids[self.pairs.values]

        new_weight = func.grid_sample(new_weight, grid, align_corners=True)
        new_weight = new_weight.reshape(self.group_channels * self.out_channels,
                                        self.group_channels * self.in_channels,
                                        *self.weight.shape[3:])

        feature_maps = feature_maps.flatten(start_dim=1, end_dim=2)
        conv = func.conv3d(feature_maps, new_weight, None, self.stride, self.padding)
        conv = conv.reshape(-1, self.group_channels, self.out_channels, *conv.shape[2:])
        return conv

    def _create_weight(self, nonlinearity='relu'):
        k = self.kernel_size

        weight = torch.zeros(self.out_channels, self.group_channels * self.in_channels, *[k] * 3, dtype=torch.float)
        nn.init.kaiming_normal_(weight, nonlinearity=nonlinearity)
        weight = weight.reshape(self.out_channels, self.group_channels, self.in_channels, *[k] * 3)

        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, group={self.group_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class GNorm(nn.Module):

    def __init__(self, channels, eps=1e-8):
        super().__init__()

        self.channels = channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(channels, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float), requires_grad=True)

    def forward(self, feature_maps):
        std, mean = torch.std_mean(feature_maps, dim=[1, -1, -2, -3], keepdim=True)
        weight = self.weight.view(1, 1, -1, 1, 1, 1)
        bias = self.bias.view(1, 1, -1, 1, 1, 1)

        return weight * (feature_maps - mean) / (std + self.eps) + bias

    def extra_repr(self):
        return f'channels={self.channels}, eps={self.eps}'


class GSEAttention(nn.Module):

    def __init__(self, channels, ratio=0.5):
        super().__init__()

        self.channels = channels
        squeeze = int(ratio * channels)
        self.se = nn.Sequential(*[
            GlobalAverage([1, -1, -2, -3]),
            nn.Linear(channels, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, channels),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        att = self.se(x).reshape(-1, 1, self.channels, 1, 1, 1)
        return att * x


class GSAAttention(nn.Module):

    def __init__(self, channels, ratio=1/8):
        super().__init__()

        self.channels = channels

        self.k = nn.Conv1d(channels, int(channels * ratio), kernel_size=1)
        self.v = nn.Conv1d(channels, int(channels * ratio), kernel_size=1)
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        # x: B x G x C x D x H x W
        # flat_x: BG x C x DHW
        flat_x = x.view(-1, x.shape[2], x.shape[3] * x.shape[4] * x.shape[5])

        # k: BG x Cr x DHW
        #    BG x DHW x Cr
        # v: BG x Cr x DHW
        k = self.k(flat_x).transpose(1, 2)
        v = self.v(flat_x)

        # kv:  BG x DHW x DHW
        # att: BG x DHW x DHW
        kv = torch.matmul(k, v)
        att = kv.softmax(dim=1)

        # q: BG x C x DHW
        q = self.q(flat_x)

        # refined: BG x C x DHW
        #          B x G x C x D x H x W
        refined = torch.matmul(q, att)
        refined = refined.view(*x.shape)

        result = self.gamma * refined + x
        return result


def create_grid(group, kernel):
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

    affine_grids = torch.einsum("ijka,rba->rijkb", grids, group)
    affine_grids = nn.Parameter(affine_grids, requires_grad=False)
    return affine_grids
