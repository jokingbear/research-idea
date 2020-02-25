from plasma.modules import *

dr = AttentionRouting(8, 16, capsules=3)
a = torch.ones(1, 8 * 32, 16, 16)
b = torch.ones(1, 3 * 16, 16, 16)

print(dr(a, b).shape)
