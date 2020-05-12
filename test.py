import torch

from plasma.training import losses


a = torch.randn(5, 8, device="cuda:0")

closs = losses.contrastive_loss_fn()
closs(a, a)
