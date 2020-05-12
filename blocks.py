from plasma.modules import *


class Conv_BN_ReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=1, dilation=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

        if act:
            self.act = nn.ReLU(inplace=True)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bottleneck_ratio, groups, att_ratio=1/16, down=False):
        super().__init__()

        bottleneck = int(out_channels * bottleneck_ratio * groups)
        self.skip = nn.Sequential(*[
            Conv_BN_ReLU(in_channels, bottleneck, kernel=1, padding=0),
            Conv_BN_ReLU(bottleneck, bottleneck, stride=2 if down else 1, groups=groups),
            Conv_BN_ReLU(bottleneck, out_channels, kernel=1, padding=0, act=False),
            SEAttention(out_channels, ratio=att_ratio)
        ])

        if in_channels != out_channels or down:
            self.identity = Conv_BN_ReLU(in_channels, out_channels, kernel=1, padding=0,
                                         stride=2 if down else 1, act=False)
        else:
            self.identity = Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        skip = self.skip(x)

        total = identity + skip
        total = self.act(total)

        return total


class Stage(nn.Sequential):

    def __init__(self, width_in, widths, bottleneck_ratio, groups, att_ratio):
        super().__init__()

        for i, width in enumerate(widths):
            self.add_module(f"block_{i}", ResBlock(width_in, width, bottleneck_ratio, groups,
                                                   att_ratio, down=i + 1 == len(widths)))
            width_in = width


class SEResNext(nn.Sequential):

    def __init__(self, stem_width, stages_widths, bottleneck_ratio, groups, att_ratio):
        super().__init__()

        self.stem = nn.Sequential(*[
            Normalization(),
            Conv_BN_ReLU(1, stem_width, stride=2),
        ])

        width = stem_width
        for i, widths in enumerate(stages_widths):
            self.add_module(f"stage_{i}", Stage(width, widths, bottleneck_ratio, groups, att_ratio))
            width = widths[-1]

        self.attention = SAModule(width[-1])


class GCNN(nn.Module):

    def __init__(self, embedding_npy, correlation_npy, n_feature, bottleneck_ratio):
        super().__init__()

        correlation = nn.Parameter(torch.tensor(correlation_npy, dtype=torch.float), requires_grad=False)

        self.graph = GraphSequential(embedding_npy, *[
            GraphLinear(300, int(n_feature * bottleneck_ratio), correlation),
            nn.LeakyReLU(0.2, inplace=True),
            GraphLinear(int(n_feature * bottleneck_ratio), n_feature, correlation_npy)
        ])

        self.bias = nn.Parameter(torch.zeros(n_feature), requires_grad=True)

    def forward(self, x):
        embedding = self.graph()
        projection = nn.functional.linear(x, embedding, self.bias)

        return projection


class CheXNeXt(nn.Sequential):

    def __init__(self, configs, embedding_npy, correlation_npy):
        super().__init__()

        features_configs = {k: configs[k] for k in configs if k != "graph_bottleneck_ratio"}
        self.features = SEResNext(**features_configs)

        feature_width = configs["stages_width"][-1][-1]
        bottleneck = configs["graph_bottleneck_ratio"]
        self.graph = GCNN(embedding_npy, correlation_npy, feature_width, bottleneck)
