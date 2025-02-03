import plasma.functional as F
import torch

from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class GeneratedResults:
    results: list[str]
    scores: list[float]


class Generator(F.AutoPipe):

    @abstractmethod
    def run(self, context_features:torch.Tensor) -> GeneratedResults:
        pass
