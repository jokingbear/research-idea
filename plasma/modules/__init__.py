from plasma.modules.blocks import \
    Normalize_Activate_Transform_Block as NAT_Block, \
    Transform_Normalize_Activate_Block as TNA_Block
from plasma.modules.commons import GlobalAverage, Reshape
from plasma.modules.group_convolution_2d import PrimaryGroupConv2d, GroupConv2d, GroupBatchNorm2d
from plasma.modules.router import DynamicRouting
