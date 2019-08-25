import tensorflow as tf

from tensorflow.keras import layers, initializers as inits
from implementations.utils import standardize_kernel_stride_dilation


class GroupConv(layers.Layer):

    def __init__(self, rank, n_group, n_filer, kernel_sizes, strides, dilations, use_bias, kernel_initializer,
                 use_native=False, **kwargs):
        super().__init__(**kwargs)

        self.rank = rank
        self.n_group = n_group
        self.n_filter = n_filer
        self.kernel_sizes = standardize_kernel_stride_dilation(rank, "kernel", kernel_sizes)
        self.strides = standardize_kernel_stride_dilation(rank, "stride", strides)
        self.dilations = standardize_kernel_stride_dilation(rank, "dilation", dilations)
        self.use_bias = use_bias
        self.kernel_initializer = inits.get(kernel_initializer)
        self.use_native = use_native

        self.kernels = None
        self.bias = None

    def build(self, input_shape):
        ks = list(self.kernel_sizes)
        gp = input_shape[-1]
        g = self.n_group
        p = gp // g
        f = self.n_filter

        self.kernels = [self.add_weight(f"weights_{i}", shape=[*ks, p, f], initializer=self.kernel_initializer)
                        for i in range(g)]

        self.bias = self.add_weight("bias", shape=[g * f], initializer="zeros") if self.use_bias else None

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.use_native:
            ws = self.kernels
        else:
            g = self.n_group
            gp = inputs.shape[-1]
            p = gp // g

            spatial_pad = [(0, 0)] * self.rank
            ws = [tf.pad(w, [*spatial_pad, (i * p, (g - 1 - i) * p), (0, 0)]) for i, w in zip(range(g), self.kernels)]

        w = tf.concat(ws, -1)
        con = tf.nn.convolution(inputs, w, self.strides, "SAME", dilations=self.dilations)
        con = con + self.bias if self.use_bias else con

        return con

    def compute_output_signature(self, input_signature):
        spatial_shape = input_signature[1:-1].as_list()
        spatial_shape = [s // stride for s, stride in zip(spatial_shape, self.strides)]
        g = self.n_group
        f = self.n_filter

        shape = (None, *spatial_shape, g * f)
        return shape

    def get_config(self):
        config = super().get_config()

        config["n_group"] = self.n_group
        config["n_filter"] = self.n_filter
        config["kernel_sizes"] = self.kernel_sizes
        config["strides"] = self.strides
        config["dilations"] = self.dilations
        config["use_bias"] = self.use_bias
        config["kernel_intializer"] = inits.serialize(self.kernel_initializer)

        return config


class GroupConv2D(GroupConv):

    def __init__(self, n_group, n_filter, kernel_sizes, strides=(1, 1),
                 dilations=None,
                 kernel_initializer="glorot_uniform",
                 use_bias=False, **kwargs):
        super().__init__(2, n_group, n_filter, kernel_sizes, strides,
                         dilations or 1, use_bias, kernel_initializer, **kwargs)


class DynamicRouting(layers.Layer):

    def __init__(self, n_group, n_iter=3, use_bias=True, **kwargs):
        super().__init__(**kwargs)

        self.n_group = n_group
        self.n_iter = n_iter
        self.use_bias = use_bias

        self.bias = None

    def build(self, input_shape):
        gf = input_shape[-1]
        g = self.n_group
        f = gf // g
        self.bias = self.add_weight("bias", shape=[f], initializer="zeros") if self.use_bias else None

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        shape = inputs.shape.as_list()
        spatial = shape[1:-1]
        gf = shape[-1]
        g = self.n_group

        inputs = tf.reshape(inputs, [-1, *spatial, g, gf // g])
        beta = tf.constant(0, dtype=tf.float32)

        for i in range(self.n_iter):
            alpha = tf.sigmoid(beta)
            v = tf.reduce_sum(inputs * alpha, axis=-2, keepdims=True)

            if i == self.n_iter - 1:
                return v[..., 0, :] + self.bias

            beta = beta + tf.reduce_sum(inputs * v, axis=-1, keepdims=True)

    def compute_output_signature(self, input_signature):
        spatial = input_signature[1:-1]
        g = self.n_group
        f = input_signature[-1] // g

        shape = (None, *spatial, f)

        return shape

    def get_config(self):
        config = super().get_config()

        config["n_group"] = self.n_group
        config["n_iter"] = self.n_iter
        config["use_bias"] = self.use_bias

        return config
