import torch.nn as nn
from torchvision import models

from plasma.modules import router


def dynamic_routing_next50(pretrained=True):
    resnext = models.resnext50_32x4d(pretrained=pretrained)
    return resnext.apply(replace_by_dynamic_routing)


def dynamic_routing_next101(pretrained=True):
    resnext = models.resnext101_32x8d(pretrained=pretrained)
    return resnext.apply(replace_by_dynamic_routing)


def attention_next50(pretrained=True):
    resnext = models.resnext50_32x4d(pretrained=pretrained)
    return resnext.apply(replace_by_attention_routing)


def apply_iters(module, iters=3):
    if isinstance(module, router.DynamicRouting2d):
        module.iters = iters


def replace_by_dynamic_routing(module):
    if isinstance(module, models.resnet.Bottleneck):
        conv3 = module.conv3
        module.conv3 = router.DynamicRouting2d(conv3.in_channels // module.conv2.groups, conv3.out_channels,
                                               groups=module.conv2.groups, iters=1, bias=False)
        module.conv3.load_state_dict(conv3.state_dict())


def replace_by_attention_routing(module):
    class AttentionBottleneck(nn.Module):

        def __init__(self, bottleneck: models.resnet.Bottleneck):
            super().__init__()

            self.bottleneck = bottleneck

            groups = bottleneck.conv2.groups
            btn = bottleneck.conv2.out_channels // groups
            out_channels = bottleneck.conv3.out_channels
            self.routing_att = router.AttentionRouting(btn, out_channels, groups=groups)

        def forward(self, x):
            identity = x

            out = self.bottleneck.conv1(x)
            out = self.bottleneck.bn1(out)
            out = self.bottleneck.relu(out)

            out = self.bottleneck.conv2(out)
            out = self.bottleneck.bn2(out)
            embedding = out
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.routing_att(embedding, out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    if isinstance(module, nn.Sequential):
        for i in range(len(module)):
            if isinstance(module[i], models.resnet.Bottleneck):
                module[i] = AttentionBottleneck(module[i])
