import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from plasma.modules import *
from torchvision.utils import make_grid


model = nn.Sequential(*[
    PrimaryGroupConv2d(1, 8, kernel_size=3, padding=1),
    GroupBatchNorm2d(8),
    nn.ReLU(inplace=True),

    GroupConv2d(8, 16, kernel_size=3, padding=1),
    GroupBatchNorm2d(16),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(2, 2),

    GroupConv2d(16, 16, kernel_size=3, padding=1),
    GroupBatchNorm2d(16),
    nn.ReLU(inplace=True),

    GroupConv2d(16, 32, kernel_size=3, padding=1),
    GroupBatchNorm2d(32),
    nn.ReLU(inplace=True),

    nn.MaxPool2d(2, 2),

    GroupConv2d(32, 32, kernel_size=3, padding=1),
    GroupBatchNorm2d(32),
    nn.ReLU(inplace=True),

    Reshape(32, -1, 7, 7),
    GlobalAverage(rank=3),
    nn.Linear(32, 3),
    # nn.Softmax(dim=-1)
])

arr = []
model.cuda(0)[0].register_forward_hook(lambda m, inp, out: arr.append(out))

a = torch.randn(1, 1, 28, 28, device="cuda:0")

with torch.no_grad():
    model.eval()(a)


f = arr[0].view(-1, 1, 28, 28)
grid = make_grid(f, nrow=12, normalize=True)

plt.imshow(grid.permute([1, 2, 0]).cpu().numpy())
plt.show()
