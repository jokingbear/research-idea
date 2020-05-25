import torch

from plasma.training import losses


a = torch.randn(5, 6)
b = torch.randn(5, 6)
l = losses.contrastive_loss_fn()

l(a, b)
