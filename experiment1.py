from keras import layers, Model
from implementation.group_conv import GroupConv2D

x = layers.Input([28, 28, 128])

y = GroupConv2D(32, 4, 3, strides=2)(x)


