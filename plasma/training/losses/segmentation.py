import numpy as np
import torch
import torch.nn as nn


class TVLoss(nn.Module):

    def __init__(self, norm=2):
        super().__init__()

        self.norm = norm

    def forward(self, x):
        rank = len(x.shape[2:])

        shift_h = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=2)
        shift_w = torch.cat([x[:, :, :, 1:], x[:, :, :, :1]], dim=3)
        shifts = [shift_h, shift_w]

        if rank > 2:
            shift_z = torch.cat([x[..., 1:], x[..., :1]], dim=-1)
            shifts.append(shift_z)

        shifts = torch.stack(shifts, dim=0)
        x = x[np.newaxis]

        if self.norm == 1:
            tv = abs(shifts - x)
        elif self.norm % 2 == 0:
            tv = (shifts - x).pow(self.norm)
        else:
            tv = abs(shifts - x).pow(self.norm)

        return tv.sum(dim=0).mean()

    def extra_repr(self):
        return f"norm={self.norm}"
