import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as func

from plasma.modules import *


a = torch.zeros(36, 36)
a[:, 17] = 1
a[17, :] = 1
a = a[np.newaxis, np.newaxis]

alpha = (np.random.rand() * 2 - 1) * (10 * np.pi / 180)

R = torch.tensor([
    [np.cos(alpha), -np.sin(alpha)],
    [np.sin(alpha), np.cos(alpha)]
], dtype=torch.float)

theta = func.pad(R, (0, 1))
inv_theta = func.pad(R.transpose(0, 1), (0, 1))

grids = func.affine_grid(theta[np.newaxis], a.shape, align_corners=True)
inv_grids = func.affine_grid(inv_theta[np.newaxis], a.shape, align_corners=True)

b = func.grid_sample(a, grids, align_corners=True)
c = func.grid_sample(b, inv_grids, align_corners=True)

_, axes = plt.subplots(ncols=3, figsize=(15, 15))

axes[0].imshow(a[0, 0])
axes[1].imshow(b[0, 0])
axes[2].imshow(c[0, 0])
plt.show()
