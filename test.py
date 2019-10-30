import torch.nn as nn

import plasma.initializations as inits

inits.standard_layers = {}

inits.kaiming_init()(nn.Conv2d(1, 2, 3))
