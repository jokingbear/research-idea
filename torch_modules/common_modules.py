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
        if self.mode == "concat":
            return torch.cat(xs, dim=1)
        else:
            final = xs[0]

            for x in xs[1:]:
                final += x

            return final

    def extra_repr(self):
        return f"mode={self.mode}"
