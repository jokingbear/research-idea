from plasma.modules import *


a = torch.ones(5, 8, 8, 8)
e = torch.ones(5, 12)

norm = GraphAdaBatchNorm(5, 12, nn.BatchNorm2d(8))
b = norm(a, e)
