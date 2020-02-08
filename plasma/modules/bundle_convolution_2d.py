import numpy as np
import torch
import torch.nn as nn

from plasma.modules import router, ChannelAttention

representation = 2
n_angle = 12
sigma = 0.5
gaussian_kernel_dict = {}


def register_gaussian_kernel(kernel, radius, n_ring):
    if kernel not in gaussian_kernel_dict:
        polar = compute_polar_to_cartesian_grid(kernel, radius, n_ring)
        center = compute_center_cartesian_grid(kernel)

        gaussian_kernel_dict[kernel] = {
            "polar": nn.Parameter(polar, requires_grad=False),
            "center": nn.Parameter(center, requires_grad=False)
        }

    return gaussian_kernel_dict[kernel]["polar"], gaussian_kernel_dict[kernel]["center"]


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
