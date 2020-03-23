import torch.nn as nn

from plasma.modules import *
from torchvision import models


a = models.resnext50_32x4d()

Frozen.freeze(a, nn.BatchNorm2d)
