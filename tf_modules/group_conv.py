import tensorflow as tf

from tensorflow.keras import layers, initializers as inits
from tf_modules.utils import standardize_kernel_stride_dilation


use_native = False


class GroupConv(layers.Layer):

    def __init__(self, rank, n_group, n_filer, kernel_sizes, strides, dilations, use_bias, kernel_initializer,
                 **kwargs):
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
        w = build_weights_for_group_conv(self.kernels, self.n_group, inputs.shape[-1])
        if self.rank == 2:
            con_op = tf.nn.conv2d
        elif self.rank == 3:
            con_op = tf.nn.conv3d
        else:
            con_op = tf.nn.convolution

        con = con_op(inputs, w, self.strides, "SAME", dilations=self.dilations)
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

    def __init__(self, n_group, n_filter, n_iter=3, use_bias=True, **kwargs):
        super().__init__(**kwargs)

        self.n_group = n_group
        self.n_filter = n_filter
        self.n_iter = n_iter
        self.use_bias = use_bias

        self.kernels = None
        self.bias = None

    def build(self, input_shape):
        ks = [1] * len(input_shape.shape[1:-1])
        gfi = input_shape[-1]
        fo = self.n_filter

        self.kernels = self.add_weight("weights", shape=[*ks, gfi, fo], initializer="he_normal")
        self.bias = self.add_weight("bias", shape=[fo], initializer="zeros") if self.use_bias else None

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        ws = build_weights_for_group_conv(self.kernels, self.n_group, inputs.shape[-1])
        con = tf.nn.convolution(inputs, ws, strides=1)

        shape = con.shape.as_list()
        spatial = shape[1:-1]
        gf = shape[-1]
        g = self.n_group

        con = tf.reshape(con, [-1, *spatial, g, gf // g])
        beta = tf.constant(0, dtype=tf.float32)

        for i in range(self.n_iter):
            alpha = tf.sigmoid(beta)
            v = tf.reduce_sum(con * alpha, axis=-2, keepdims=True)

            if i == self.n_iter - 1:
                v = v[..., 0, :]
                return v + self.bias if self.use_bias else v

            beta = beta + tf.reduce_sum(con * v, axis=-1, keepdims=True)

    def compute_output_signature(self, input_signature):
        spatial = input_signature[1:-1]
        f = self.n_filter

        shape = (None, *spatial, f)

        return shape

    def get_config(self):
        config = super().get_config()

        config["n_group"] = self.n_group
        config["n_filter"] = self.n_filter
        config["n_iter"] = self.n_iter
        config["use_bias"] = self.use_bias

        return config


def build_weights_for_group_conv(weights, n_group, gp):
    shape = weights.shape
    g = n_group

    if use_native:
        ws = tf.reshape(weights, [*shape[:-2], g, -1, shape[-1]])
    else:
        g = n_group
        p = gp // g
        rank = len(weights[0].shape[:-2])
        spatial_pad = [(0, 0)] * rank
        ws = [tf.pad(w, [*spatial_pad, (i * p, (g - 1 - i) * p), (0, 0)]) for i, w in zip(range(g), weights)]

    return tf.concat(ws, -1)


def build_weights_for_routing(weights, rank, n_group, fi, fo):
    kshape = weights.shape.to_list()[:-2]
    g = n_group
    ws = weights

    if use_native:
        raise NotImplemented()
    else:
        ws = tf.reshape(ws, [*kshape, g, fi, fo])
        ws = tf.transpose(ws, [rank] + list(range))

        pass
