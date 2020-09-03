from plasma.modules import *


a = torch.randn(1, 4992, 8, 8, dtype=torch.float, device="cuda:0")
b = attentions.DSAModule(4992).cuda()

c = b(a)
