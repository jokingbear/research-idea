import torch

from plasma import models

m = models.ResCap(None)

a = torch.ones(1, 1, 320, 384)
m(a)
