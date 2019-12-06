import torch
import torch.nn as nn


class GeneratorDiscriminator(nn.Module):

    def __init__(self, generator, discriminator):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, g_grad, g_kwargs, d_kwargs):
        if g_grad:
            imgs = self.generator(**g_kwargs)
        else:
            with torch.no_grad():
                imgs = self.generator(**g_kwargs)

        imgs = imgs if type(imgs) in {tuple, list} else [imgs]
        scores = self.discriminator(*imgs, **d_kwargs)

        return scores
