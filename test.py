import torch
import torch.nn as nn

from torch_modules import blocks, commons

a = commons.MergeModule(mode="add")

b = [torch.ones(1, 2), torch.ones(1, 2) * 2, torch.ones(1, 2) * 3]

a(*b)
