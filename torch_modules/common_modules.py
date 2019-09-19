import torch
import torch.nn as nn


class GlobalAverage(nn.Module):

    def forward(self, x):
        rank = len(x.shape[2:])
        axes = tuple([2 + i for i in range(rank)])

        return torch.mean(x, dim=axes)


class Reshape(nn.Module):

    def __init__(self, *shape):
        super().__init__()

        self.shape = shape

    def forward(self, x):
        return x.reshape([-1, *self.shape])
