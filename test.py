import torch

from plasma.modules import *

a = torch.ones(1, 3, 5, 5, dtype=torch.int) * 255

b = ImagenetNorm(from_gray=False)(a)

from albumentations import RandomBrightnessContrast