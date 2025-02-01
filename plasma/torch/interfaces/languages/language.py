import torch
import torch.nn as nn

from .tokenizer import Tokenizer
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class Language(nn.Module):
    tokenizer:Tokenizer
    nfeatures:int

    @abstractmethod
    def forward(self, 
                tokens:torch.Tensor, 
                attentions:torch.Tensor=None,
                generative_indices:torch.Tensor=None,
                context_embeddings:torch.Tensor=None,
                cache=None)->torch.Tensor:
        pass
