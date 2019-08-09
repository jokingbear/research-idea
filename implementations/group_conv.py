import tensorflow as tf

from keras import layers, backend as K, activations, initializers as inits
from keras.utils import conv_utils as utils
from implementations.utils import standardize_kernel_stride_dilation

tf_native = False


class GroupConv2D(layers.Layer):

    def __init__(self, n_group, n_filter, kernel_sizes, strides=(1, 1), padding="same",
                 dilation_rate=None,
                 activation=None,
                 kernel_initializer="glorot_uniform", **kwargs):
        self.n_group = n_group
        self.n_filter = n_filter
        self.kernel_sizes = standardize_kernel_stride_dilation(2, "kernel_size", kernel_sizes)
        self.strides = standardize_kernel_stride_dilation(2, "strides", strides)
        self.padding = padding
        self.dilation_rate = standardize_kernel_stride_dilation(2, "dilation_rate", dilation_rate or 1)
        self.activation = activations.get(activation)
        self.kernel_initializer = inits.get(kernel_initializer)

        self.w = None
        self.b = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        if tf_native:
            _build_native(self, input_shape)
        else:
            _build_non_native(self, input_shape)

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


class GroupRouting2D(layers.Layer):

    def __init__(self, n_group, n_filter, kernel_size, strides=1, padding="same", n_iter=3,
                 kernel_initializer="he_normal", **kwargs):
        self.n_group = n_group
        self.n_filter = n_filter
        self.kernel_sizes = standardize_kernel_stride_dilation(2, "kernel", kernel_size)
        self.strides = standardize_kernel_stride_dilation(2, "stride", strides)
        self.padding = padding
        self.n_iter = n_iter
        self.kernel_initializer = inits.get(kernel_initializer)

        self.w = None
        self.b = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        _build_native(self, input_shape) if tf_native else _build_non_native(self, input_shape)

        self.b = self.add_weight("bias", [self.n_filter], initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        conv = K.conv2d(inputs, self.w, self.strides, padding=self.padding)

        return _route(conv, self.n_group, self.n_iter, self.b)

    def compute_output_shape(self, input_shape):
        _, h, w, gp = input_shape

        return None, h, w, self.n_filter


def _build_native(layer, input_shape):
    kh, kw = layer.kernel_sizes
    gp = input_shape[-1]
    g = layer.n_group
    p = gp // g
    f = layer.n_filter

    ws = [layer.add_weight(f"weights_{i}", [kh, kw, p, f],
                           initializer=layer.kernel_initializer) for i in range(g)]

    # (k, k, p, gf)
    layer.w = K.concatenate(ws)


def _build_non_native(layer, input_shape):
        kh, kw = layer.kernel_sizes
        gp = input_shape[-1]
        g = layer.n_group
        p = gp // g
        f = layer.n_filter

        ws = []
        for i in range(g):
            w = layer.add_weight(f"weights_{i}", [kh, kw, p, f], initializer=layer.kernel_initializer)

            # (kh, kw, gp, f)
            w = tf.pad(w, [[0, 0], [0, 0], [i * p, (g - 1 - i) * p], [0, 0]])

            ws.append(w)

        # (k, k, gp, gf)
        layer.w = K.concatenate(ws)


def _route(inputs, n_group, n_iter, bias):
    shape = inputs.shape.as_list()
    spatial = shape[1:-1]
    gf = shape[-1]
    g = n_group

    inputs = tf.reshape(inputs, [-1] + spatial + [g, gf // g])
    beta = tf.constant(0, dtype=tf.float32)

    for i in range(n_iter):
        alpha = tf.sigmoid(beta)
        v = tf.reduce_sum(inputs * alpha, axis=-2, keepdims=True)

        if i == n_iter - 1:
            return v + bias

        beta = beta + tf.reduce_sum(inputs * v, axis=-1, keepdims=True)
