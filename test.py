import torch
import torch.nn as nn
import torchvision.models as models

import plasma.hub as hub
import siamphys


from wavenext import WaveNext

a = siamphys.SiamPhys(30, 30)

b = torch.rand(1, 3, 810, 32, 32)

with torch.no_grad():
    c = a(b, b)
