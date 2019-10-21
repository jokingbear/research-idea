import torch_modules.blocks as blocks

from torch import nn


def get_standard_layer():
    return {
        blocks.conv_layer, blocks.router_layer
    }


def kaiming_init(a=0, zero_bias=False, exceptions=None, verbose=1):
    exceptions = exceptions or set()

    def kaiming(m):
        if type(m) in get_standard_layer() - exceptions:
            nn.init.kaiming_normal_(m.weight, a)

            if m.bias is not None and zero_bias:
                nn.init.zeros_(m.bias)

            print("kaiming init ", m) if verbose else None

    return kaiming
