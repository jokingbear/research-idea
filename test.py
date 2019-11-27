import torch
import torch.nn as nn

from plasma.modules import DynamicRouting, GlobalAverage


class BN_ReLU_Conv(nn.Sequential):

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=1, groups=1):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_features)
        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, groups=groups)


class BN_ReLU_Routing(nn.Sequential):

    def __init__(self, in_features, out_features, groups=32, iters=1):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_features * groups)
        self.activation = nn.ReLU(inplace=True)
        self.conv = DynamicRouting(in_features, out_features, groups, iters)


class RoutingBlock(nn.Sequential):

    def __init__(self, in_features, bottleneck, out_features, groups=32, iters=1, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1

        self.embed1 = BN_ReLU_Conv(in_features, bottleneck * groups, kernel_size=1, padding=0)
        self.embed2 = BN_ReLU_Conv(bottleneck * groups, bottleneck * groups, groups=groups, stride=stride)
        self.routing = BN_ReLU_Routing(bottleneck, out_features, groups, iters)


class DenseBlock(nn.Module):

    def __init__(self, in_features, bottleneck, routing_features, out_features, blocks, groups, iters):
        super().__init__()

        self.down_sampling = nn.AvgPool2d(2, 2)
        self.down_routing = RoutingBlock(in_features, bottleneck, routing_features, groups, iters, downsample=True)

        self.routing_modules = nn.ModuleList([
            RoutingBlock(in_features + routing_features * (i + 1), bottleneck, routing_features, groups, iters)
            for i in range(blocks)])

        self.finalize = BN_ReLU_Conv(in_features + routing_features * (blocks + 1), out_features)

    def forward(self, x):
        down_sample = self.down_sampling(x)
        down_routing = self.down_routing(x)

        x = torch.cat([down_sample, down_routing], dim=1)
        for m in self.routing_modules:
            y = m(x)
            x = torch.cat([x, y], dim=1)

        return self.finalize(x)


class DenseCap(nn.Sequential):

    def __init__(self, n_channels=1, f0=64, groups=32, iters=1):
        super().__init__()

        bottleneck = 4
        routing_features = 8
        blocks = 4

        self.con0 = nn.Sequential(*[
            nn.Conv2d(n_channels, f0, kernel_size=7, stride=2, padding=3),
            BN_ReLU_Conv(f0, f0),
        ])  # 64 x 256 x 256

        # 128 x 128 x 128
        self.con1 = DenseBlock(f0, bottleneck, routing_features, 2 * f0, blocks, groups, iters)  # 128 x 128 x 128

        # 256 x 64 x 64
        self.con2 = DenseBlock(2 * f0, 2 * bottleneck, 2 * routing_features, 4 * f0, blocks * 2, groups, iters)

        # 512 x 32 x 32
        self.con3 = DenseBlock(4 * f0, 4 * bottleneck, 4 * routing_features, 8 * f0, blocks * 4, groups, iters)

        # 1024 x 16 x 16
        self.con4 = DenseBlock(8 * f0, 8 * bottleneck, 8 * routing_features, 16 * f0, blocks * 4, groups, iters)

        # 2048 x 8 x 8
        self.con5 = DenseBlock(16 * f0, 8 * bottleneck, 8 * routing_features, 32 * f0, blocks * 2, groups, iters)

        self.classifier = nn.Sequential(*[
            nn.BatchNorm2d(32 * f0),
            nn.ReLU(inplace=True),
            GlobalAverage(),
            nn.Linear(32 * f0, 14),
            nn.Sigmoid()
        ])


a = DenseCap()
a(torch.ones(1, 1, 384, 384))
