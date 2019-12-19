import numpy as np
import torch
import torch.nn as nn

n_angle = 12
sigma = 0.5
n_radius = 5


class PrimaryGroupConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, n_radius, n_angle),
                                   requires_grad=True)
        self.center = nn.Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None

        self.center_gauss = nn.Parameter(compute_center_cartesian_grid(kernel_size), requires_grad=False)
        self.polar_gauss = nn.Parameter(compute_polar_to_cartesian_grid(kernel_size), requires_grad=False)
        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.center)

    def forward(self, x):
        # create rotated version of kernel
        idc = np.array(range(n_angle))
        rotation_shuffle = (idc[np.newaxis, :] - idc[:, np.newaxis] + n_angle) % n_angle
        ws = [self.weight[..., r] for r in rotation_shuffle]
        w = torch.stack(ws, dim=1).view(-1, self.in_channels, n_radius, n_angle)

        # sample from non origin part
        polar_gauss = self.polar_gauss
        w = torch.einsum("xyrt,oirt->oixy", [polar_gauss, w])

        # sample from center of kernel
        center_gauss = self.center_gauss
        center = center_gauss[np.newaxis, np.newaxis] * self.center[..., np.newaxis, np.newaxis]
        center = center.repeat(1, n_angle, 1, 1, 1).view(-1, self.in_channels, self.kernel, self.kernel)

        # calculate correlation
        w = w + center
        bias = self.bias.repeat(n_angle) if self.bias is not None else None
        con = torch.conv2d(x, w, bias, self.stride, self.padding)

        return con

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel={self.kernel}, " \
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

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, n_angle, n_radius, n_angle),
                                   requires_grad=True)
        self.center = nn.Parameter(torch.zeros(out_channels, in_channels, n_angle), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None

        self.center_gauss = nn.Parameter(compute_center_cartesian_grid(kernel_size), requires_grad=False)
        self.polar_gauss = nn.Parameter(compute_polar_to_cartesian_grid(kernel_size), requires_grad=False)
        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.center)

    def forward(self, x):
        # create permutation indices
        idc = np.array(range(n_angle))
        rotation_shuffle = (idc[None, :] - idc[:, None] + n_angle) % n_angle

        # sample from center
        center_gauss = self.center_gauss[np.newaxis, np.newaxis, np.newaxis]  # 111XY
        center = self.center[..., np.newaxis, np.newaxis]  # OIR11
        centers = [center_gauss * center[:, :, r, ...] for r in rotation_shuffle]
        center = torch.stack(centers, dim=1).view(-1, self.in_channels * n_angle, self.kernel, self.kernel)

        # sample from non origin
        polar_gauss = self.polar_gauss
        ws = [self.weight[:, :, r, ...][..., r] for r in rotation_shuffle]
        w = torch.stack(ws, dim=1).view(-1, self.in_channels * n_angle, n_radius, n_angle)
        w = torch.einsum("xyrt,oirt->oixy", [polar_gauss, w])

        # calculate correlation
        w = w + center
        bias = self.bias.repeat(n_angle) if self.bias is not None else None
        con = torch.conv2d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)

        return con

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel={self.kernel}, " \
               f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
               f"bias={self.bias is not None}"


class GroupBatchNorm2d(nn.BatchNorm3d):

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x.view(x.shape[0], -1, n_angle, h, w)
        x = super().forward(x).view(x.shape[0], -1, h, w)
        return x


def compute_polar_to_cartesian_grid(kernel):
    with torch.no_grad():
        # generate polar coordinate
        radius = torch.linspace(start=0, end=kernel, steps=n_radius + 1)[1:]
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
        gauss = gauss / gauss.sum(dim=[-1, -2], keepdim=True)

        return gauss


def compute_center_cartesian_grid(kernel):
    # generate kernel cartesian coordinate
    xs = torch.linspace(-kernel / 2, kernel / 2, steps=kernel)
    ys = xs

    # calculate propagation from origin to kernel coordinate
    dist2 = xs[:, np.newaxis].pow(2) + ys[np.newaxis, :].pow(2)
    gauss = (-dist2 / 2 / sigma ** 2).exp()
    gauss = gauss / gauss.sum()

    return gauss

# TODO: add dilation and groups
