import torch
import torch.nn as nn


class HFlip(nn.Module):

    def __init__(self, model, inverse_output=False):
        super().__init__()

        self.inverse_output = inverse_output
        self.model = model

    def forward(self, x):
        flip = x.flip(dims=[-1])
        x = torch.cat([x, flip], dim=0)
        out = self.model(x)
        out = out.view(2, -1, *out.shape[1:])

        out0 = out[0]
        out1 = out[1]

        if self.inverse_output:
            out1 = out1.flip(dims=[-1])

        return (out0 + out1) / 2
