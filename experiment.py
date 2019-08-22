import tensorflow as tf
import numpy as np


@tf.function
def con(x, w):
    return tf.nn.conv2d(x, w, strides=(2, 2), padding="SAME", dilations=[6, 6])


x_val = np.random.rand(1, 28, 28, 1)
w_val = np.random.rand(3, 3, 1, 32)

c = con(x_val, w_val)
