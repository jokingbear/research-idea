import torch
import plasma.modules as modules

att = modules.SEAttention(15, dims=[1, -1, -2, -3])
b = torch.randn(5, 12, 15, 10, 10, 10)

with torch.no_grad():
    c = att(b)
