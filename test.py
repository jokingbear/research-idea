import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
import time

from physnext_2 import PhysNext

a = PhysNext(27, 30, in_channels=24)

b = torch.rand(1, 3, 810, 96, 96)

with torch.no_grad():
    start = time.time()
    c = a(b)
    end = time.time()
    print(end - start)
