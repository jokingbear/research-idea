from plasma.modules.configs import *


class Transform_Normalize_Activate_Block(nn.Sequential):

    def __init__(self, out_features, transformer, normalizer, activator):
        super().__init__()

        self.transformer = transformer
        self.normalizer = normalizer or default_normalization(out_features)
        self.activator = activator or activation_layer


class Normalize_Activate_Transform_Block(nn.Sequential):

    def __init__(self, out_features, normalizer, activator, transformer):
        super().__init__()

        self.normalizer = normalizer or default_normalization(out_features)
        self.activator = activator or activation_layer
        self.transformer = transformer
