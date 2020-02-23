from torchvision import models
from plasma.modules import router


def dynamic_routing_next50():
    resnext = models.resnext50_32x4d(pretrained=True)
    return resnext.apply(replace_by_dynamic_routing)


def dynamic_routing_next101():
    resnext = models.resnext101_32x8d(pretrained=True)
    return resnext.apply(replace_by_dynamic_routing)


def apply_iters(module, iters=3):
    if isinstance(module, router.DynamicRouting2d):
        module.iters = iters


def replace_by_dynamic_routing(module):
    if isinstance(module, models.resnet.Bottleneck):
        conv3 = module.conv3
        module.conv3 = router.DynamicRouting2d(conv3.in_channels // module.conv2.groups, conv3.out_channels,
                                               groups=module.conv2.groups, iters=1, bias=False)
        module.conv3.load_state_dict(conv3.state_dict())
