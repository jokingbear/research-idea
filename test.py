import torchvision.models as models

from plasma.modules import *


a = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')

torch.save(a.state_dict(), "resnext101_32x16_swsl.model")
