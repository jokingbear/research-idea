import torch
import torch.hub as hub

from plasma.training import losses
from torchvision import models


#model = hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl').cuda(0)
model = models.densenet121().cuda()

a = torch.ones(48, 3, 256, 256, device="cuda:0")
model(a)
