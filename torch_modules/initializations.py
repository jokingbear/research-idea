import torch_modules.blocks as blocks

from torch import nn


standard_layers = {
    blocks.ConvBlock, blocks.RoutingBlock
}


def kaiming_init(a=0, zero_bias=False, exceptions=None, verbose=1):
    exceptions = exceptions or set()

    def kaiming(m):
        if type(m) in standard_layers - exceptions:
            nn.init.kaiming_normal_(m.weight, a)

            if m.bias is not None and zero_bias:
                nn.init.zeros_(m.bias)

            print("kaiming init ", m) if verbose else None

    return kaiming
