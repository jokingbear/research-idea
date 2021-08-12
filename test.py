import numpy as np
import pandas as pd

from plasma.modules import *

a = np.random.rand(36, 36, 36)
b = np.rot90(a, axes=(1, 2)).copy()

at = torch.tensor(a, dtype=torch.float, device='cuda')[np.newaxis, np.newaxis]
bt = torch.tensor(b, dtype=torch.float, device='cuda')[np.newaxis, np.newaxis]

grids7 = groups.create_grid(7)
grids3 = groups.create_grid(3)

model = nn.Sequential(*[
    groups.GConvPrime(1, 5, grids7, padding=3),
    groups.GPool(),
    groups.GConv(5, 8, grids3, padding=1),
    GlobalAverage([1, -1, -2, -3])
]).cuda()

with torch.no_grad():
    print(model(at))
    print(model(bt))
