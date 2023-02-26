import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from plasma.modules import GlobalAverage, ImagenetNorm
from torchvision.models.feature_extraction import create_feature_extractor


class TimeDiffNormalization(nn.Module):

    def forward(self, X):
        # X: B T C H W
        D = torch.diff(X, dim=1)
        C = X[:, 1:] + X[:, :-1]

        # BT C HW
        results = D / C
        std, mean = torch.std_mean(results, dim=1, keepdim=True)
        results = (results - mean) / std
        results = results.flatten(end_dim=1)
        return results


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
        
        return mask / mean_mask


class AppearanceNext(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        backbone = models.convnext_tiny().features

        self.imagenet = ImagenetNorm()
        self.extractor = create_feature_extractor(backbone, return_nodes=['1', '3', '5', '7'])
        self.attentions = [SelfAttention(in_channel) for in_channel in [96, 192, 384, 768]]

    def forward(self, X):
        # X: B T C HW -> B C HW
        X = X.mean(dim=1)
        X = self.imagenet(X)

        feature_maps = self.extractor(X)

        results = []
        for att, k in zip(self.attentions, feature_maps):
            att_map = att(feature_maps[k])
            print(att_map.shape)
            results.append(att_map)
        
        return results


class MotionNext(nn.Sequential):

    def __init__(self, duration, fps, nclass=1):
        super().__init__()

        nframe = np.ceil(duration * fps)
        nframe = int(nframe)

        self.diff = TimeDiffNormalization()
        self.features, nfeatures = self.process_backbone(nframe - 1)

        self.head = nn.Sequential(*[
            GlobalAverage(),
            nn.Linear(nfeatures, nclass)
        ])
    
    def forward(self, X, attention_maps):
        B, T = X.shape[:2]

        diffX = self.diff(X)

        results = diffX
        print(results.shape)
        for att_map, feature_block in zip(attention_maps, self.features):
            results = feature_block(results)
            results = results.view(B, T - 1, *results.shape[1:])
            results = results * att_map[:, np.newaxis]
            results = results.flatten(end_dim=1)
            print(results.shape)
        
        results = self.head(results)
        results = results.unflatten(dim=0, sizes=[B, T - 1])

        return results

    def process_backbone(self, ndframe):
        temp_shift = TemporalShift(ndframe)
        backbone = models.convnext_tiny(weights='IMAGENET1K_V1').features
        
        new_backbone = []
        for i, module in enumerate(backbone.children()):
            if i % 2 != 0:
                for cn_block in module.children():
                    name = type(cn_block).__name__

                    if name == 'CNBlock':
                        old_block = cn_block.block
                        cn_block.block = nn.Sequential(temp_shift, *old_block)
            
            new_backbone.append(module)

        blocks = [
            nn.Sequential(new_backbone[0], new_backbone[1]),
            nn.Sequential(new_backbone[2], new_backbone[3]),
            nn.Sequential(new_backbone[4], new_backbone[5]),
            nn.Sequential(new_backbone[6], new_backbone[7]),
        ]
        return nn.ModuleList(blocks), 768


class PhysNext(nn.Module):

    def __init__(self, duration, fps, nclass=1):
        super().__init__()

        self.appearance = AppearanceNext()
        self.motion = MotionNext(duration, fps, nclass)
    
    def forward(self, X):
        attention_maps = self.appearance(X)
        
        return self.motion(X, attention_maps)
