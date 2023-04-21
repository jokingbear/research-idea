import torch
import torch.nn as nn
import torchvision.ops as vision_ops

import numpy as np
import plasma.modules as modules



class ConvNorm(nn.Sequential):
    
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride)
        self.permute = modules.Permute(0, 2, 1)
        self.norm = nn.LayerNorm(out_channel)
        self.final = modules.Permute(0, 2, 1)


class CNBlock(nn.Module):
    
    def __init__(self, channels, p, layer_scale=1e-6):
        super().__init__()
        
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale)
        
        self.block = nn.Sequential(*[
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels),
            modules.Permute(0, 2, 1),
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.LayerNorm(channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            modules.Permute(0, 2, 1),
        ])
        
        self.stochastic_depth = vision_ops.StochasticDepth(p, mode='row')
    
    def forward(self, inputs):
        residuals = self.layer_scale * self.block(inputs)
        residuals = self.stochastic_depth(residuals)
        results = inputs + residuals
        return results


class DNBlock(nn.Module):
    
    def __init__(self, channels, p, layer_scale=1e-6):
        super().__init__()
        
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale)
        
        self.block = nn.Sequential(*[
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=7, padding=3, groups=channels),
            modules.Permute(0, 2, 1),
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, channels * 4),
            nn.LayerNorm(channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            modules.Permute(0, 2, 1),
        ])
        
        self.stochastic_depth = vision_ops.StochasticDepth(p, mode='row')
        self.combine = nn.Sequential(*[
            modules.Permute(0, 2, 1),
            nn.LayerNorm(2 * channels),
            nn.Conv1d(2 * channels, channels, kernel_size=7, padding=3)
        ])
    
    def forward(self, inputs):
        residuals = self.layer_scale * self.block(inputs)
        residuals = self.stochastic_depth(residuals)
        results = self.combine(inputs) + residuals
        return results


class ABPNext(nn.Module):
    
    def __init__(self, in_channel=1, encoder_blocks=[(2, 32), (2, 64), (3, 128), (4, 256)], p=0.1):
        super().__init__()
        
        stem_channel = encoder_blocks[0][1]
        self.stem = ConvNorm(in_channel, stem_channel, 4, 4)
        
        self._create_encoders(encoder_blocks, p)
        self._create_decoders(encoder_blocks)

        self.head = nn.Sequential(*[
            modules.Permute(0, 2, 1),
            nn.LayerNorm(stem_channel),
            modules.Permute(0, 2, 1),
            nn.Conv1d(stem_channel, 1, kernel_size=3, padding=1)
        ])
    
    def forward(self, inputs):
        stem = self.stem(inputs)
        
        skips = []
        results = stem
        for i, e in enumerate(self.encoders):
            results = e(results)

            if i < len(self.encoders) - 1:
                skips.append(results)
        
        skips = skips[::-1]
        for up, decoder, skip in zip(self.ups, self.decoders, skips):
            results = up(results)
            results = torch.cat([results, skip], dim=1)
            results = decoder(results)

    
    def _create_encoders(self, encoder_blocks, p):
        total_blocks = sum([nblock for nblock, _ in encoder_blocks], 0)
        prev_channels = encoder_blocks[0][1]
        encoders = []
        i = 0
        for j, (nblock, channels) in enumerate(encoder_blocks):
            resolution_blocks = []
            
            if j != 0:
                down = nn.Sequential(*[
                    modules.Permute(0, 2, 1),
                    nn.LayerNorm(prev_channels),
                    modules.Permute(0, 2, 1),
                    nn.Conv1d(prev_channels, channels, kernel_size=2, stride=2),
                ])
                
                resolution_blocks.append(down)
            
            for _ in range(nblock):
                block = CNBlock(channels, i * p / (total_blocks - 1))
                resolution_blocks.append(block)
                i += 1
            
            resolution_blocks = nn.Sequential(*resolution_blocks)
            encoders.append(resolution_blocks)
        
        self.encoders = nn.ModuleList(encoders)

    def _create_decoders(self, encoder_blocks):
        decoder_blocks = encoder_blocks[::-1][1:]

        prev_channels = encoder_blocks[-1][1]
        decoders = []
        ups = []
        for _, channels in decoder_blocks:
            up = nn.Sequential(*[
                modules.Permute(0, 2, 1),
                nn.LayerNorm(prev_channels),
                modules.Permute(0, 2, 1),
                nn.ConvTranspose1d(prev_channels, channels)
            ])

            res = DNBlock(channels, 0.5)

            ups.append(up)
            decoders.append(res)
        
        self.ups = nn.ModuleList(ups)
        self.decoders = nn.ModuleList(decoders)
