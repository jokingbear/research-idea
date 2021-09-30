import torch
import torch.nn as nn

import plasma.modules as modules

grids = modules.s4.create_grid(7)
grids2 = modules.s4.create_grid(3)

att = nn.Sequential(*[
    modules.s4.S4Prime(1, 10, grids, padding=3),
    modules.s4.S4Norm(10),
    nn.LeakyReLU(negative_slope=0.2, inplace=True),
    modules.s4.S4Pool(kind='average'),
    modules.s4.S4Linear(10, 16),
    nn.LeakyReLU(negative_slope=0.2, inplace=True),
    modules.s4.S4Conv(4, 16, grids2, padding=1, partition=4),
    modules.s4.S4Linear(16, 8),
    modules.GlobalAverage(dims=[-1, -2, -3, -4]),
]).cuda()

b = torch.randn(1, 1, 10, 10, 10, device='cuda:0')
b = torch.cat([
    b,
    b.rot90(k=1, dims=[-1, -2])
], dim=0)

with torch.no_grad():
    c = att(b)
    print(c)
