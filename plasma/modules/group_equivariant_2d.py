import numpy as np
import torch
import torch.nn as nn

n_angle = 12
sigma = 0.5


class PrimaryGroupConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding

        radius, n_ring = compute_radius_ring(kernel_size)
        self.radius = radius
        self.n_ring = n_ring
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, n_ring, n_angle), requires_grad=True)
        self.center = nn.Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None

        gauss_grid = compute_polar_to_cartesian_grid(kernel_size, radius, n_ring)
        center_grid = compute_center_cartesian_grid(kernel_size)
        self.polar_gauss = nn.Parameter(gauss_grid, requires_grad=False)
        self.center_gauss = nn.Parameter(center_grid, requires_grad=False)

        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.center)

    def forward(self, x):
        # create rotated version of kernel
        idc = np.array(range(n_angle))
        ws = [self.weight[..., (idc - a + n_angle) % n_angle] for a in range(n_angle)]
        w = torch.stack(ws, dim=1).view(-1, self.in_channels, self.n_ring, n_angle)

        # sample from non origin part
        w_radial = torch.einsum("xyrt,oirt->oixy", [self.polar_gauss, w])

        # center part
        w_center = self.center[..., np.newaxis, np.newaxis] * self.center_gauss[np.newaxis, np.newaxis]
        w_center = torch.stack([w_center] * n_angle, dim=1).view(-1, self.in_channels, self.kernel, self.kernel)

        # calculate correlation
        w = w_radial + w_center
        bias = torch.stack([self.bias] * n_angle, dim=1).flatten() if self.bias is not None else None
        con = torch.conv2d(x, w, bias, self.stride, self.padding)

        return con

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel={self.kernel}, " \
               f"raidus={self.radius}, n_ring={self.n_ring}, " \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}"


class GroupConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        radius, n_ring = compute_radius_ring(kernel_size)
        self.radius = radius
        self.n_ring = n_ring
        self.weight = nn.Parameter(torch.zeros(groups * out_channels, in_channels, n_angle, n_ring, n_angle),
                                   requires_grad=True)
        self.center = nn.Parameter(torch.zeros(groups * out_channels, in_channels, n_angle), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(groups * out_channels), requires_grad=True) if bias else None

        polar_gauss = compute_polar_to_cartesian_grid(kernel_size, radius, n_ring)
        center_gauss = compute_center_cartesian_grid(kernel_size)
        self.polar_gauss = nn.Parameter(polar_gauss, requires_grad=False)
        self.center_gauss = nn.Parameter(center_gauss, requires_grad=False)
        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.center)

    def forward(self, x):
        # sample from non origin
        idc = np.array(range(n_angle))
        polar_gauss = self.polar_gauss.to(x.device)
        ws = [self.weight[:, :, (idc - i + n_angle) % n_angle, ...][..., (idc - i + n_angle) % n_angle]
              for i in range(n_angle)]
        w_radial = torch.stack(ws, dim=1).view(-1, self.in_channels * n_angle, self.n_ring, n_angle)
        w_radial = torch.einsum("xyrt,oirt->oixy", [polar_gauss, w_radial])

        # sample from center
        w_center = torch.stack([self.center[..., (idc - i + n_angle) % n_angle] for i in range(n_angle)], dim=1)
        w_center = w_center.view(-1, self.in_channels * n_angle)
        w_center = w_center[..., np.newaxis, np.newaxis] * self.center_gauss[np.newaxis, np.newaxis]

        # calculate correlation
        w = w_radial + w_center
        bias = torch.stack([self.bias] * n_angle, dim=1).flatten() if self.bias is not None else None
        con = torch.conv2d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)

        return con

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel={self.kernel}, " \
               f"radius={self.radius}, n_ring={self.n_ring}, " \
               f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
               f"bias={self.bias is not None}"


class GroupMapping(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(groups * out_channels, in_channels, 1, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(groups * out_channels), requires_grad=True) if bias else None
    
    def forward(self, x):
        return so2_mapping(x, self.weight, self.bias, self.groups)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, groups={self.groups}, " \
               f"bias={self.bias is not None}"


class GroupBatchNorm2d(nn.BatchNorm3d):

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x.view(x.shape[0], -1, n_angle, h, w)
        x = super().forward(x).view(x.shape[0], -1, h, w)
        return x


def compute_polar_to_cartesian_grid(kernel, radius, n_ring):
    # generate polar coordinate
    radius = torch.linspace(start=0, end=radius, steps=n_ring + 1)[1:]
    angles = torch.linspace(start=0, end=2 * np.pi, steps=n_angle + 1)[:-1]

    # calculate cartesian coordinate
    source_x = radius[:, np.newaxis] * angles.cos()[np.newaxis, :]
    source_y = radius[:, np.newaxis] * angles.sin()[np.newaxis, :]

    # generate kernel cartesian coordinate
    xs = torch.linspace(-kernel / 2, kernel / 2, steps=kernel)
    ys = xs

    # calculate euclidean distance
    x_diff = xs[:, np.newaxis, np.newaxis, np.newaxis] - source_x[np.newaxis, np.newaxis]
    y_diff = ys[np.newaxis, :, np.newaxis, np.newaxis] - source_y[np.newaxis, np.newaxis]
    dist2 = x_diff.pow(2) + y_diff.pow(2)

    # calculate the propagation of source to kernel coordinate
    gauss = (-dist2 / 2 / sigma ** 2).exp()
    # gauss = gauss / gauss.sum(dim=[-1, -2], keepdim=True)

    return gauss


def compute_center_cartesian_grid(kernel):
    xs = torch.linspace(-kernel / 2, kernel / 2, steps=kernel)
    ys = xs

    # calculate propagation from origin to kernel coordinate
    dist2 = xs[:, np.newaxis].pow(2) + ys[np.newaxis, :].pow(2)
    gauss = (-dist2 / 2 / sigma ** 2).exp()
    # gauss = gauss / gauss.sum()

    return gauss


def compute_radius_ring(kernel):
    radius = kernel / np.sqrt(2)
    n_ring = kernel
    return radius, n_ring


def so2_mapping(x, weight, bias=None, groups=1):
    h, w = x.shape[-2:]
    in_channels = x.shape[1] // n_angle
    
    x = x.view(-1, in_channels, n_angle, h, w)
    weight = weight if len(weight.shape[2:]) == 3 else weight[..., np.newaxis]
    con = torch.conv3d(x, weight, bias, groups=groups)

    return con

# TODO: add extra repr
# TODO: add dilation and groups
