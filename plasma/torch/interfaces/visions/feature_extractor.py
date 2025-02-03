import torch

from .vision import Vision
from abc import abstractmethod


class Features:

    def __init__(self, globals: list[torch.Tensor], intermedidates: list[torch.Tensor]):
        self.globals = globals
        self.intermediates = intermedidates

    def __repr__(self):
        tab = ' ' * 2

        return (
            'Features\n'
            f'{tab}globals={[f.shape for f in self.globals]},\n'
            f'{tab}intermediates={[fi.shape for fi in self.intermediates]}\n'
        )


class FeatureExtractor(Vision):

    @abstractmethod
    def forward(self, inputs:torch.Tensor) -> Features:
        pass
