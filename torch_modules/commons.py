import torch
import torch.nn as nn


class GlobalAverage(nn.Module):

    def __init__(self, rank=2):
        super().__init__()

        self.axes = list(range(2, 2 + rank))

    def forward(self, x):
        return torch.mean(x, dim=self.axes)

    def extra_repr(self):
        return f"axes={self.axes}"


class Reshape(nn.Module):

    def __init__(self, *shape):
        super().__init__()

        self.shape = shape

    def forward(self, x):
        return x.reshape([-1, *self.shape])

    def extra_repr(self):
        return f"shape={self.shape}"


class MergeModule(nn.Module):

    def __init__(self, mode="concat"):
        super().__init__()
        self.mode = mode

    def forward(self, *xs):
        if len(xs) == 1:
            return xs[0]

        if self.mode == "concat":
            return torch.cat(xs, dim=1)
        elif len(xs) == 2:
            return xs[0] + xs[1]
        else:
            return torch.stack(xs, dim=0).sum(dim=0)

    def extra_repr(self):
        return f"mode={self.mode}"
