import tensorflow as tf

from keras import layers, backend as K, activations, initializers as inits, regularizers as regs
from keras.utils import conv_utils as utils


class GroupConv2D(layers.Layer):

    def __init__(self, n_groups, n_filters, kernel_size, stride=1, padding="same",
                 activation=None,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None, **kwargs):
        self.n_groups = n_groups
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = (stride, stride)
        self.padding = padding
        self.activation = activations.get(activation)
        self.kernel_initializer = inits.get(kernel_initializer)
        self.kernel_regularizer = regs.get(kernel_regularizer)

        super().__init__(**kwargs)

    def build(self, input_shape):
        k = self.kernel_size
        gp = input_shape[-1]
        g = self.n_groups
        p = gp // g
        f = self.n_filters

        ws = []
        for i in range(g):
            w = self.add_weight(f"weights_{i}", [k, k, p, f],
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer)

            # (k, k, gp, f)
            w = tf.pad(w, [[0, 0], [0, 0], [i * p, (g - 1 - i) * p], [0, 0]])

            ws.append(w)

        # (k, k, gp, gf)
        self.w = K.concatenate(ws)
        self.b = self.add_weight("bias", [g * f], initializer="zeros")

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        h, w, _ = input_shape[1:]
        k = self.kernel_size
        
        h = utils.conv_output_length(h, k, self.padding, self.strides[0])
        w = utils.conv_output_length(w, k, self.padding, self.strides[1])
        
        return None, h, w, self.n_groups * self.n_filters

    def call(self, inputs, **kwargs):
        return self.activation(K.conv2d(inputs, self.w, self.strides, self.padding) + self.b)
