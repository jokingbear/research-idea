import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class HorizontalFlip(nn.Module):

    def forward(self, x):
        return x.flip(dims=[-1])


class Zoom(nn.Module):

    def __init__(self, scale=0.1, interpolation="bilinear"):
        """
        :param scale: scale to zoom, positive for zooming in, negative for zooming out
        :param interpolation: how to resize zoomed image
        """
        super().__init__()

        assert -1 < scale < 0 or 0 < scale < 1, "scale doesn't lie in range (-1, 0) U (0, 1)"
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, x):
        h, w = x.shape[-2:]

        if self.scale > 0:
            zoom_h = int(np.round(h * self.scale))
            zoom_w = int(np.round(w * self.scale))
            x = x[..., zoom_h:-zoom_h, zoom_w:-zoom_w]
        else:
            pad_h = -int(np.round(h * self.scale))
            pad_w = -int(np.round(w * self.scale))
            x = func.pad(x, [pad_w] * 2 + [pad_h] * 2)

        return func.interpolate(x, size=(h, w), mode=self.interpolation, align_corners=True)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, interpolation={self.interpolation}"


class Compose(nn.Module):

    def __init__(self, main_module, aug_modules):
        """
        :param main_module: main computation module
        :param aug_modules: test time augmentation modules
        """
        super().__init__()

        assert len(aug_modules) > 0, "must have at least 1 augmentation module"

        self.aug_modules = nn.ModuleList(aug_modules)
        self.main_module = main_module

    def forward(self, x):
        x = [x] + [aug_module(x) for aug_module in self.aug_modules]
        x = torch.stack(x, dim=1).flatten(start_dim=0, end_dim=1)
        results = self.main_module(x)
        return results.view(-1, 1 + len(self.aug_modules), *results.shape[1:])
