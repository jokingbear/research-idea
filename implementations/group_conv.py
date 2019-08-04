import tensorflow as tf

from keras import layers, backend as K, activations, initializers as inits, regularizers as regs
from keras.utils import conv_utils as utils

tf_native = False


class GroupConv2D(layers.Layer):

    def __init__(self, n_group, n_filter, kernel_sizes, strides=(1, 1), padding="same",
                 dilation_rate=None,
                 activation=None,
                 kernel_initializer="glorot_uniform", **kwargs):
        self.n_group = n_group
        self.n_filter = n_filter
        self.kernel_sizes = self._standardize_kernel_stride_dilation("kernel_size", kernel_sizes)
        self.strides = self._standardize_kernel_stride_dilation("strides", strides)
        self.padding = padding
        self.dilation_rate = self._standardize_kernel_stride_dilation("dilation_rate",
                                                                      dilation_rate if dilation_rate else 1)
        self.activation = activations.get(activation)
        self.kernel_initializer = inits.get(kernel_initializer)

        super().__init__(**kwargs)

    def build(self, input_shape):
        if tf_native:
            self._build_native(input_shape)
        else:
            self._build_non_native(input_shape)

        g = self.n_group
        f = self.n_filter

        self.b = self.add_weight("bias", [g * f], initializer="zeros")

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        h, w, _ = input_shape[1:]
        kh, kw = self.kernel_sizes
        
        h = utils.conv_output_length(h, kh, self.padding, self.strides[0])
        w = utils.conv_output_length(w, kw, self.padding, self.strides[1])
        
        return None, h, w, self.n_group * self.n_filter

    def call(self, inputs, **kwargs):
        strides = (1,) + self.strides + (1,)
        dilations = (1,) + self.dilation_rate + (1,)
        padding = self.padding.upper()
        conv = tf.nn.conv2d(inputs, self.w, strides, padding, dilations=dilations)
        conv = conv + self.b
        conv = self.activation(conv)

        return conv

    def _standardize_kernel_stride_dilation(self, check_type, value):
        if type(value) is int:
            return value, value
        elif (type(value) is list or type(value) is tuple) and len(value) == 2:
            return value

        raise TypeError(f"{value} is not a valid {check_type}")

    def _build_native(self, input_shape):
        kh, kw = self.kernel_sizes
        gp = input_shape[-1]
        g = self.n_group
        p = gp // g
        f = self.n_filter

        ws = [self.add_weight(f"weights_{i}", [kh, kw, p, f],
                              initializer=self.kernel_initializer) for i in range(g)]

        # (k, k, p, gf)
        self.w = K.concatenate(ws)

    def _build_non_native(self, input_shape):
        kh, kw = self.kernel_sizes
        gp = input_shape[-1]
        g = self.n_group
        p = gp // g
        f = self.n_filter

        ws = []
        for i in range(g):
            w = self.add_weight(f"weights_{i}", [kh, kw, p, f], initializer=self.kernel_initializer)

            # (kh, kw, gp, f)
            w = tf.pad(w, [[0, 0], [0, 0], [i * p, (g - 1 - i) * p], [0, 0]])

            ws.append(w)

        # (k, k, gp, gf)
        self.w = K.concatenate(ws)