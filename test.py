import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from physnext_3 import PhysNext

a = PhysNext(5, 30)

b = torch.rand(2, 150, 3, 96, 96)

with torch.no_grad():
    c = a(b)
