from keras import layers, Model
from implementations import conv_routing as cr, group_conv as gc

x = layers.Input([512, 512, 128])

y = cr.GroupRouting(32, 18, 3)(x)

model = Model(x, y)

w = model.get_weights()

a = 8