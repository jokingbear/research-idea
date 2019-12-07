import numpy as np
import torch

w = torch.ones(64, 32, 5, 12)

x_grid = torch.linspace(start=-3 / 2, end=3 / 2, steps=3)
y_grid = x_grid

radials = torch.linspace(start=1, end=5, steps=5)
angles = torch.linspace(start=0, end=11 * 2 * np.pi / 12, steps=12)

xs = radials.view(-1, 1) * angles.cos().view(1, -1)
ys = radials.view(-1, 1) * angles.sin().view(1, -1)

x_dist = (x_grid.view(-1, 1, 1) - xs.view(1, 5, 12)).pow(2)
y_dist = (y_grid.view(-1, 1, 1) - ys.view(1, 5, 12)).pow(2)

dist = x_dist.view(-1, 1, 5, 12) + y_dist.view(1, -1, 5, 12)

gauss = (-dist / 2 / 5 ** 2).exp()
gauss = gauss / gauss.sum(dim=[-1, -2], keepdim=True)

g_w = torch.einsum("xyrt,oirt->oixy", [gauss, w])
