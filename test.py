import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from physnext_2 import PhysNext

a = PhysNext(3, 30)

b = torch.rand(1, 90, 3, 96, 96)

with torch.no_grad():
    c = a(b)

