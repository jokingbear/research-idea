import numpy as np

from implementations import normalization_layers as nlayers, group_conv as gc
from tensorflow.python.keras import layers, Model, models


x = layers.Input([28, 28, 32])


con = gc.GroupConv2D(4, 16, 3, strides=2)(x)
con = nlayers.GroupNorm(4)(con)
con = gc.GroupRouting2D(4, 32, 3)(con)
con = nlayers.GroupNorm(4)(con)

model = Model(x, con)
model.save("test")

