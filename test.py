from plasma.modules import *


a = torch.ones(5, 128, 128)
b = torch.ones(5, 128, 128, 3)

m = ImageToTensor()
print(m(a).shape)
print(m(b).shape)
