import torch
import torch.nn as nn

rank = 2
con_op = torch.conv2d
conv_layer = nn.Conv2d
pooling_layer = nn.AvgPool2d
activation_layer = nn.ReLU(inplace=True)


def default_normalization(f):
    return nn.BatchNorm2d(f)
