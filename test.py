import torch

from physnext import PhysNext


a = torch.rand(2, 60, 3, 128, 128)
b = PhysNext(0, 1)

with torch.no_grad():
    c = b(a)