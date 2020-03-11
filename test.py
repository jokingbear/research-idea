from plasma.modules import *


class Conv_BN_ReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=1, attention=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels * groups, out_channels * groups, kernel, stride, padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_channels * groups)

        if attention:
            self.att = CBAM(out_channels * groups)

        self.act = nn.ReLU(inplace=True)


class Routing_BN(nn.Sequential):

    def __init__(self, in_channels, out_channels, capsules=1, groups=32, iters=1):
        super().__init__()

        self.router = DynamicRouting2d(in_channels, out_channels, capsules, groups, iters)
        self.norm = nn.BatchNorm2d(out_channels)


class CapsuleBlock(nn.Module):

    def __init__(self, in_channels, bottleneck, capsules=1, groups=32, iters=1, downsample=False):
        super().__init__()

        self.skip = nn.Sequential(*[
            Conv_BN_ReLU(in_channels, bottleneck * groups, kernel=1, padding=0),
            Conv_BN_ReLU(bottleneck, bottleneck, stride=2 if downsample else False, groups=groups),
            Routing_BN(bottleneck, in_channels, capsules, groups, iters)
        ])

        self.down_sample = downsample
        self.identity = nn.MaxPool2d(2, 2) if downsample else Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = self.skip(x)
        identity = self.identity(x)

        if self.down_sample:
            skip = self.act(skip)
            return torch.cat([identity, skip], dim=1)
        else:
            result = identity + skip
            result = self.act(result)
            return result


class ResCap(nn.Module):

    def __init__(self, capsules=1, iters=1):
        super().__init__()

        self.encoder = dynamic_routing_next50()
