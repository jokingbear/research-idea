import numpy as np
import torch
import torch.nn as nn

angle_kernel = 12
sigma = 0.5
radius_kernel = 5


class PrimaryGroupConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, radius_kernel, angle_kernel),
                                   requires_grad=True)
        self.center = nn.Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None

        self.center_gauss = nn.Parameter(compute_center_cartesian_grid(kernel_size), requires_grad=False)
        self.polar_gauss = nn.Parameter(compute_polar_to_cartesian_grid(kernel_size), requires_grad=False)
        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.center)

    def forward(self, x):
        # create rotated version of kernel
        idc = np.array(range(angle_kernel))
        rotation_shuffle = (idc[None, :] - idc[:, None] + angle_kernel) % angle_kernel
        ws = [self.weight[..., r] for r in rotation_shuffle]
        w = torch.stack(ws, dim=1).view(-1, self.in_channels, radius_kernel, angle_kernel)

        # sample from non origin part
        polar_gauss = self.polar_gauss
        w = torch.einsum("xyrt,oirt->oixy", [polar_gauss, w])

        # sample from center of kernel
        center_gauss = self.center_gauss
        center = center_gauss[np.newaxis, np.newaxis] * self.center[..., np.newaxis, np.newaxis]
        center = center.repeat(1, angle_kernel, 1, 1, 1).view(-1, self.in_channels, self.kernel, self.kernel)

        # calculate correlation
        w = w + center
        bias = self.bias.repeat(angle_kernel) if self.bias is not None else None
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

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, angle_kernel, radius_kernel, angle_kernel),
                                   requires_grad=True)
        self.center = nn.Parameter(torch.zeros(out_channels, in_channels, angle_kernel), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True) if bias else None

        self.center_gauss = nn.Parameter(compute_center_cartesian_grid(kernel_size), requires_grad=False)
        self.polar_gauss = nn.Parameter(compute_polar_to_cartesian_grid(kernel_size), requires_grad=False)
        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.center)

    def forward(self, x):
        # create permutation indices
        idc = np.array(range(angle_kernel))
        rotation_shuffle = (idc[None, :] - idc[:, None] + angle_kernel) % angle_kernel

        # sample from center
        center_gauss = self.center_gauss[np.newaxis, np.newaxis, np.newaxis]
        center = self.center[..., np.newaxis, np.newaxis]
        centers = [center_gauss * center[:, :, r, ...] for r in rotation_shuffle]
        center = torch.stack(centers, dim=1).view(-1, self.in_channels * angle_kernel, self.kernel, self.kernel)

        # sample from non origin
        polar_gauss = self.polar_gauss
        ws = [self.weight[:, :, r, ...][..., r] for r in rotation_shuffle]
        w = torch.stack(ws, dim=1).view(-1, self.in_channels * angle_kernel, radius_kernel, angle_kernel)
        w = torch.einsum("xyrt,oirt->oixy", [polar_gauss, w])

        # calculate correlation
        w = w + center
        bias = self.bias.repeat(angle_kernel) if self.bias is not None else None
        con = torch.conv2d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)

        return con

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel={self.kernel}, " \
               f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, " \
               f"bias={self.bias is not None}"


class GroupBatchNorm2d(nn.BatchNorm3d):

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x.view(x.shape[0], -1, angle_kernel, h, w)
        x = super().forward(x).view(x.shape[0], -1, h, w)
        return x


def compute_polar_to_cartesian_grid(kernel):
    radius = torch.linspace(start=0, end=kernel, steps=radius_kernel + 1)[1:]
    angles = torch.linspace(start=0, end=2 * np.pi, steps=angle_kernel + 1)[:-1]

    source_x = radius[:, np.newaxis] * angles.cos()[np.newaxis, :]
    source_y = radius[:, np.newaxis] * angles.sin()[np.newaxis, :]

    xs = torch.linspace(-kernel / 2, kernel / 2, steps=kernel)
    ys = xs

    x_dist2 = (xs.view(-1, 1, 1, 1) - source_x.view(1, 1, radius_kernel, angle_kernel)).pow(2)
    y_dist2 = (ys.view(1, -1, 1, 1) - source_y.view(1, 1, radius_kernel, angle_kernel)).pow(2)
    dist2 = x_dist2 + y_dist2

    gauss = (-dist2 / 2 / sigma ** 2).exp()
    gauss = gauss / gauss.sum(dim=[-1, -2], keepdim=True)

    return gauss


def compute_center_cartesian_grid(kernel):
    xs = torch.linspace(-kernel / 2, kernel / 2, steps=kernel)
    ys = xs

    dist2 = xs.view(-1, 1).pow(2) + ys.view(1, -1).pow(2)
    gauss = (-dist2 / 2 / sigma ** 2).exp()
    gauss = gauss / gauss.sum()

    return gauss

# TODO: add dilation and groups
