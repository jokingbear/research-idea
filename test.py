import torch
import torch.nn as nn

from torch_modules import commons


class Generator(nn.Sequential):

    def forward(self, x=None):
        x = x or torch.randn(32, 128, device="cuda:0")

        return super().forward(x)


g = Generator(
    nn.Linear(128, 7 * 7 * 256),
    nn.BatchNorm2d(7 * 7 * 256),
    nn.ReLU(),
    commons.Reshape(256, 7, 7),
    # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
    # nn.BatchNorm2d(128),
    # nn.ReLU(),
    # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
    # nn.BatchNorm2d(64),
    # nn.ReLU(),
    # nn.Conv2d(64, 1, 1),
    # nn.Tanh()
).cuda(0)
