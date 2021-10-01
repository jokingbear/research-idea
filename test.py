import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from default_model import Segmentator

b = torch.randn(1, 1, 64, 64, 64, device='cuda:0')
b = torch.cat([
    b,
    b.rot90(k=1, dims=[-1, -2])
], dim=0)

a = Segmentator().cuda()

with torch.no_grad():
    c = a(b)

tmp = c[-1].mean(dim=2).cpu().numpy()

_, axes = plt.subplots(nrows=4, ncols=2, figsize=(5, 20))

for i in range(4):
    axes[i, 0].imshow(tmp[0, i, 0])
    axes[i, 1].imshow(tmp[1, i, 0])
plt.show()
