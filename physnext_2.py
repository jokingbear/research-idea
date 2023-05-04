import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from plasma.modules import GlobalAverage


class TimeDiffNormalization(nn.Module):

    def __init__(self, layer_norm=True, resolution=None):
        super().__init__()

        if layer_norm:
            if resolution is None:
                self.norm = models.convnext.LayerNorm2d(3)
            else:
                self.norm = nn.LayerNorm([3, *resolution])
        else:
            self.norm = nn.BatchNorm2d(3)
    
    def forward(self, X):
        # X: B T C H W
        D = torch.diff(X, dim=1)

        # BT C HW
        D = D.flatten(end_dim=1)
        D = self.norm(D)
        return D


class TemporalShift(nn.Module):

    def __init__(self, nframe, fold_div=3):
        super().__init__()

        self.nframe = nframe
        self.fold_div = fold_div

    def forward(self, X):
        # X: BT C HW -> B T C HW
        X = X.view(-1, self.nframe, *X.shape[1:])

        C = X.shape[2]

        size = C // self.fold_div

        results = torch.zeros_like(X)

        results[:, :-1, :size] = X[:, 1:, :size]
        results[:, 1:, size:2 * size] = X[:, :-1, size: 2 * size]
        results[:, :, 2 * size:] = X[:, :, 2 * size:]

        results = results.flatten(end_dim=1)
        return results
    
    def extra_repr(self) -> str:
        return f'nframe={self.nframe}, fold_div={self.fold_div}'


class SelfAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.mask = nn.Sequential(*[
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        ])

    def forward(self, X):
        mask = self.mask(X)
        mean_mask = mask.mean(dim=[-1, -2], keepdim=True)
        
        return mask / mean_mask * X


class PhysNext(nn.Sequential):

    def __init__(self, duration, fps, nclass=1):
        super().__init__()

        nframe = np.ceil(duration * fps)
        nframe = int(nframe)

        self.diff = TimeDiffNormalization()
        self.features, nfeatures = self.process_backbone(nframe - 1)
        
        self.head = nn.Sequential(*[
            GlobalAverage(dims=[1, -1]),
            nn.Linear(nfeatures, nclass),
        ])
    
    def forward(self, X):
        B, T = X.shape[:2]

        diffX = self.diff(X)

        features = self.features(diffX)
        features = features.reshape(B, T - 1, features.shape[1], -1)

        results = self.head(features)
        return results

    def process_backbone(self, ndframe):
        temp_shift = TemporalShift(ndframe)
        backbone = models.convnext_tiny(weights='IMAGENET1K_V1').features
        
        new_backbone = []
        out_channels = 3
        for i, module in enumerate(backbone.children()):
            if i % 2 != 0:
                for cn_block in module.children():
                    name = type(cn_block).__name__

                    if name == 'CNBlock':
                        old_block = cn_block.block
                        out_channels = old_block[0].out_channels
                        cn_block.block = nn.Sequential(temp_shift, *old_block)
                        
                module.attention = SelfAttention(out_channels)
            
            new_backbone.append(module)

        return backbone, 768
 