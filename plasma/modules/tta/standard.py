import torch
import torch.nn as nn
import torch.nn.functional as func

import numpy as np


class HorizontalFlip(nn.Module):

    def forward(self, x):
        return x.flip(dims=[-1])


class Zoom(nn.Module):

    def __init__(self, scale=0.1, interpolation="bilinear"):
        super().__init__()

        assert scale != 0, "scale must not be 0"
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, x):
        h, w = x.shape[-2:]

        if self.scale > 0:
            zoom_h = np.round(h * self.scale)
            zoom_w = np.round(w * self.scale)
            x = x[..., zoom_h:-zoom_h, zoom_w:-zoom_w]
        else:
            pad_h = -np.round(h * self.scale)
            pad_w = -np.round(w * self.scale)
            x = func.pad(x, [pad_w] * 2 + [pad_h] * 2)

        return func.interpolate(x, size=(h, w), mode=self.interpolation, align_corners=True)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, interpolation={self.interpolation}"


class Compose(nn.Module):

    def __init__(self, main_module, *ttas):
        super().__init__()

        assert len(ttas) > 0, "must have at least 1 tta"

        self.ttas = nn.ModuleList(ttas)
        self.main_module = main_module

    def forward(self, x):
        results = [self.main_module(x)]

        for aug_module in self.ttas:
            aug = aug_module(x)
            results.append(self.main_module(aug))

        return torch.stack(results, dim=0)
