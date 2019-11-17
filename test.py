import torch.nn as nn

from plasma import blocks


class Identity(nn.Module):

    def forward(self, x):
        return x


class Encoder(nn.Sequential):

    def __init__(self, img_channels=1, f0=64, b0=4, groups=32, iters=1):
        super().__init__()

        self.con0 = nn.Sequential(*[
            blocks.ConvBlock(img_channels, f0)
        ])

        self.con1 = nn.Sequential(*[
            blocks.ResidualBlock(f0, b0, groups, iters, down_sample=True),
            blocks.ConvBlock(2 * f0, 2 * f0),
            blocks.ResidualBlock(2 * f0, b0, groups, iters)
        ])

        self.con2 = nn.Sequential(*[
            blocks.ResidualBlock(2 * f0, 2 * b0, groups, iters, down_sample=True),
            blocks.ConvBlock(4 * f0, 4 * f0),
            blocks.ResidualBlock(4 * f0, 2 * b0, groups, iters)
        ])

        self.con3 = nn.Sequential(*[
            blocks.ResidualBlock(4 * f0, 4 * b0, groups, iters, down_sample=True),
            blocks.ConvBlock(8 * f0, 8 * f0),
            blocks.ResidualBlock(8 * f0, 4 * b0, groups, iters)
        ])


class Decoder(nn.Sequential):

    def __init__(self, f0, img_channels=1, b0=None, groups=32, iters=1):
        super().__init__()

        self.decon2 = nn.Sequential(*[
            blocks.DeconBlock(f0, f0 // 2),
            blocks.ConvBlock(f0 // 2, f0 // 2),
            blocks.ResidualBlock(f0 // 2, b0, groups, iters) if b0 else Identity()
        ])

        self.decon1 = nn.Sequential(*[
            blocks.DeconBlock(f0 // 2, f0 // 4),
            blocks.ConvBlock(f0 // 4, f0 // 4),
            blocks.ResidualBlock(f0 // 4, b0 // 2, groups, iters) if b0 else Identity()
        ])

        self.decon0 = nn.Sequential(*[
            blocks.DeconBlock(f0 // 4, f0 // 8),
            blocks.ConvBlock(f0 // 8, f0 // 8),
            nn.Conv2d(f0 // 8, img_channels, kernel_size=1),
            nn.Tanh()
        ])


class Autoencoder(nn.Sequential):

    def __init__(self, groups=32, iters=1):
        super().__init__()

        self.encoder = Encoder(f0=64, b0=4, groups=groups, iters=iters)
        self.decoder = Decoder(f0=512, b0=None, groups=groups, iters=iters)
