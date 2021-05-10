import torch
import torch.nn as nn


a = torch.rand(5, 8, 3, 3, 3)


mean, std = torch.std_mean(a, dim=[1, 2, 3, 4], keepdim=True)
