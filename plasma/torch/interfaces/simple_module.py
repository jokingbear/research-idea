import torch
import torch.nn as nn

from abc import abstractmethod


class SimpleModule(nn.Module):

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass
