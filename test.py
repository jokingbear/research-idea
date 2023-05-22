import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
import time


from wavenext import WaveNext

a = WaveNext()

b = torch.rand(1, 1, 16 * 30)

with torch.no_grad():
    start = time.time()
    c = a(b)
    end = time.time()
    print(end - start)
