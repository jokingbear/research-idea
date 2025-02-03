import torch

from .vision import Vision
from abc import abstractmethod


class Features:
    globals: list[torch.Tensor]
    intermediates: list[torch.Tensor]

    def __repr__(self):
        tab = ' ' * 2

        return (
            'Features(\n'
            f'{tab}globals={[f.shape for f in self.globals]},\n'
            f'{tab}intermediates={[fi.shape for fi in self.intermediates]}'
        )

class FeatureExtractor(Vision):

    @abstractmethod
    def forward(self, inputs:torch.Tensor) -> Features:
        pass
