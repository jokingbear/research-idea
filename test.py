import torch
import numpy as np

from torch_modules.models import ResCap


corr_matrix = np.random.randn(14, 14)
a = ResCap(corr_matrix).cuda(0)

b = torch.randn(32, 1, 256, 256).cuda(0)

tmp = a(b)
