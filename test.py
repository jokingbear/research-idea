import torch
import torch.nn as nn


class temp(nn.Module):

    def forward(self, x, alpha=None):
        alpha = alpha or 1
        print(alpha)
        return x + alpha


b = nn.DataParallel(temp()).cuda(0)

a = torch.ones(1, 2, 3, dtype=torch.float, device="cuda:0")

b(a, alpha=5)
