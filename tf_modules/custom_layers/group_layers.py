import tensorflow as tf
import tensorflow.nn as nn
import numpy as np

from keras import layers
from tf_modules.utils import standardize_kernel_stride_dilation


class GroupConv(layers.Layer):

    def __init__(self, rank, n_group, n_filter, kernel, stride=1, dilation=1,
                 use_bias=True, kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)

        self.rank = rank
        self.n_group = n_group
        self.n_filter = n_filter
        self.kernel_sizes = standardize_kernel_stride_dilation(rank, "kernel sizes", kernel)
        self.strides = standardize_kernel_stride_dilation(rank, "strides", stride)
        self.dilations = standardize_kernel_stride_dilation(rank, "dilations", dilation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        gfi = input_shape[-1]
        g = self.n_group
        fi = gfi // g
        fo = self.n_filter
        ks = self.kernel_sizes
        kernel_shape = [*ks, fi, g * fo]

        self.kernel = self.add_weight("kernel", kernel_shape, initializer=self.kernel_initializer)
        self.bias = self.add_weight("bias", [fo * g], initializer="zeros") if self.use_bias else None

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        w = self._reshape_kernel()
        strides = (1, *self.strides, 1)
        dilations = (1, *self.dilations, 1)
        padding = "SAME"

        if self.rank == 2:
            con_op = nn.conv2d
        elif self.rank == 3:
            con_op = nn.conv3d
        else:
            con_op = nn.convolution

        con = con_op(inputs, w, strides, padding, dilations=dilations)

        if self.use_bias:
            con = nn.bias_add(con, self.bias)

        return con

    def compute_output_shape(self, input_shape):
        spatial_shape = np.array(input_shape[1:-1])
        strides = np.array(self.strides)
        f = self.n_filter
        g = self.n_group

        spatial_shape = spatial_shape // strides

        shape = (None, *spatial_shape, f * g)
        return shape

    def _reshape_kernel(self):
        w = self.kernel

        shape = w.shape.as_list()
        ks = shape[:-2]
        g = self.n_group
        fo = self.n_filter
        fi = int(w.shape[-2])

        w = tf.reshape(w, shape=[*ks, -1, g, fo])
        axes = [*range(len(ks) + 3)]
        w = tf.transpose(w, perm=[axes[-2], *axes[:-3], axes[-3], axes[-1]])

        ws = [tf.pad(w[i], [(0, 0)] * len(ks) + [(i * fi, (g - i - 1) * fi), (0, 0)]) for i in range(g)]
        ws = tf.concat(ws, axis=-1)

        return ws


class DynamicRouting(GroupConv):

    def __init__(self, rank, n_group, n_filter, n_iter, kernel_initializer="he_normal", **kwargs):
        super().__init__(rank, n_group, n_filter, kernel=1,
                         use_bias=False, kernel_initializer=kernel_initializer, **kwargs)

        self.n_iter = n_iter

    def build(self, input_shape):
        gfi = input_shape[-1]
        g = self.n_group
        fi = gfi // g
        fo = self.n_filter
        ks = self.kernel_sizes
        kernel_shape = [*ks, g * fi, fo]

        self.kernel = self.add_weight("kernel", kernel_shape, initializer=self.kernel_initializer)
        self.bias = self.add_weight("bias", [fo], initializer="zeros") if self.use_bias else None

        self.built = True

    def call(self, inputs, **kwargs):
        con = super().call(inputs)

        spatial_shape = con.shape.as_list()[1:-1]
        g = self.n_group
        f = self.n_filter

        con = tf.reshape(con, [-1, *spatial_shape, g, f])
        beta = 0.

        for i in range(self.n_iter):
            alpha = tf.sigmoid(beta)
            v = tf.reduce_sum(alpha * con, axis=-2, keepdims=True)

            if i == self.n_iter - 1:
                return v[..., 0, :]

            beta = beta + tf.reduce_sum(v * con, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        shape = super().compute_output_shape(input_shape)[:-1]

        shape = (*shape, self.n_filter)

        return shape

    def _reshape_kernel(self):
        w = self.kernel

        shape = w.shape.as_list()
        ks = shape[:-2]
        g = self.n_group
        fo = self.n_filter
        fi = int(w.shape[-2]) // g

        w = tf.reshape(w, shape=[*ks, g, -1, fo])
        axes = [*range(len(ks) + 3)]
        w = tf.transpose(w, perm=[axes[-3], *axes[:-3], axes[-2], axes[-1]])

        ws = [tf.pad(w[i], [(0, 0)] * len(ks) + [(i * fi, (g - i - 1) * fi), (0, 0)]) for i in range(g)]
        ws = tf.concat(ws, axis=-1)

        return ws


class GroupConv2D(GroupConv):

    def __init__(self, n_group, n_filter, kernel, stride=1, dilation=1,
                 use_bias=True, kernel_initializer="he_normal", **kwargs):
        super().__init__(2, n_group, n_filter, kernel, stride, dilation, use_bias, kernel_initializer, **kwargs)


class DynamicRouting2D(DynamicRouting):
    def __init__(self, n_group, n_filter, n_iter, kernel_initializer="he_normal", **kwargs):
        super().__init__(2, n_group, n_filter, n_iter, kernel_initializer, **kwargs)
