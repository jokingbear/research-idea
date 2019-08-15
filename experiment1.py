import numpy as np

from implementations import group_conv as gc
from keras import Model, layers


gc.tf_native = True

x = layers.Input([28, 28, 64])
y = gc.GroupConv2D(4, 8, 3)(x)

model = Model(x, y)

model.predict(np.random.rand(1, 28, 28, 64))

