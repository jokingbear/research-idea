import torch
import torch.nn as nn

from torch_modules import blocks, commons

a = torch.ones(1, 64, 28, 28)

b = blocks.SEBlock(64)

b(a)
