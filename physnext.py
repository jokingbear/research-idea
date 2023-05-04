import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.feature_extraction import create_feature_extractor
from plasma.modules import ImagenetNorm, GlobalAverage


class PhysNext(nn.Module):

    def __init__(self, n_class=1) -> None:
        super().__init__()

        self.motion_rep = MotionRepresentation()
        self.appearance_encoder = AppearanceEncoder()
        self.motion_encoder = MotionEncoder()
        self.head = nn.Linear(self.motion_encoder.nfeatures, n_class)
    
    def forward(self, X):
        attention_masks = self.appearance_encoder(X[:, 1:])
        D = self.motion_rep(X)
        features = self.motion_encoder(D, attention_masks)
        features = features.mean(dim=1)

        return self.head(features)


class MotionRepresentation(nn.Module):

    def forward(self, X):

        delta = X[:, 1:] - X[:, :-1]
        return delta


class AppearanceEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        feature_maps = ['features.1', 'features.3']

        self.normalize = ImagenetNorm()
        self.extractor = create_feature_extractor(backbone, feature_maps)

        self.att_projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d( 96, 1, kernel_size=1), nn.Sigmoid()),
            nn.Sequential(nn.Conv2d(192, 1, kernel_size=1), nn.Sigmoid()),
        ])
    
    def forward(self, X):
        # X: B T C HW
        # flattened_X: BT C HW
        flattened_X = X.flatten(end_dim=1)
        flattened_X = self.normalize(flattened_X)

        feature_maps = self.extractor(flattened_X)
        attention_maps = [self.get_attention_mask(feature_maps[m], p) for m, p in zip(feature_maps, self.att_projs)]

        return attention_maps

    def get_attention_mask(self, feature_map, projection):
        mask = projection(feature_map)
        mean = mask.mean(dim=[-1, -2], keepdim=True)

        return mask / mean


class MotionEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        features = models.convnext_tiny(weights='IMAGENET1K_V1').features

        self.normalize = ImagenetNorm()
        self.first_block = features[:2]
        self.second_block = features[2:4]
        self.third_block = nn.Sequential(features[4:6], GlobalAverage(), nn.LayerNorm(384))
        self.nfeatures = 384

    def forward(self, D, masks):
        #D: B T C HW
        B, T = D.shape[:2]

        # BT C HW
        flattened_D = D.flatten(end_dim=1)
        flattened_D = self.normalize(flattened_D)
        first_block = self.first_block(flattened_D)
        first_block = first_block * masks[0]

        second_block = self.second_block(first_block)
        second_block = second_block * masks[1]

        # B T C
        return self.third_block(second_block).unflatten(dim=0, sizes=[B, T])
