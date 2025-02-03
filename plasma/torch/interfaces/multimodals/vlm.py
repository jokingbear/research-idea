import torch
import torch.nn as nn

from ..visions import Vision
from ..languages import Language
from ..simple_module import SimpleModule


class VLM(nn.Module):
    vision: Vision
    projector: SimpleModule
    language: Language

    def forward(self, images:torch.Tensor, 
                tokens: torch.Tensor, attentions:torch.Tensor=None, 
                cache=None):
        vision_features = self.vision.forward(images)
        projected_vision_features = self.projector.forward(vision_features)
        language_features = self.language.forward(tokens, attentions, 
                                                  context_embeddings=projected_vision_features, 
                                                  cache=cache)     
        return language_features
