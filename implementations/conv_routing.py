import tensorflow as tf

from keras import layers, backend as K


class ConvRouting(layers.Layer):

    def __init__(self, n_group, n_iter=3, beta_initializer="zeros", **kwargs):
        super().__init__(**kwargs)

        self.n_group = n_group
        self.n_iter = n_iter
        self.beta_initializer = beta_initializer
        self.beta = None

    def call(self, inputs, **kwargs):
        shape = K.int_shape(inputs)
        g = self.n_group
        c = shape[-1] // g
        inputs = K.reshape(inputs, (-1,) + shape[1:-1] + (g, c))

        beta = K.zeros([1, 1, 1, g, 1])
        for i in range(self.n_iter):
            alpha = K.sigmoid(beta)
            pass

    def _routing(self, i, inputs, beta):
        alpha = K.sigmoid(beta)
        v = K.sum(alpha * inputs, axis=-2, keepdims=True)

        return tf.cond(i < self.n_iter - 1, lambda: (v, beta + K.sum(v * inputs, axis=(1, 2, -1))),
                       lambda: (v, beta))


class GroupRouting(layers.Layer):

    def __init__(self, n_group, n_filter, kernel, stride=1, padding="same", n_iter=3,
                 kernel_initializer="he_normal", **kwargs):
        self.n_group = n_group
        self.n_filter = n_filter
        self.kernel_sizes = kernel, kernel
        self.strides = stride, stride
        self.padding = padding
        self.n_iter = n_iter
        self.kernel_initializer = kernel_initializer

        super().__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, gp = input_shape
        g = self.n_group
        p = gp // g
        f = self.n_filter

        kh, kw = self.kernel_sizes

        w = self.add_weight("weights", [kh, kw, gp, f], initializer="he_normal")
        w = tf.reshape(w, [kw, kw, g, -1, f])
        w = tf.transpose(w, [2, 0, 1, 3, 4])

        ws = []
        for i in range(g):
            w_i = tf.pad(w[i], [(0, 0), (0, 0), (i * p, (g - i - 1) * p), (0, 0)])

            ws.append(w_i)

        self.w = tf.concat(ws, axis=-1)
        self.b = self.add_weight("bias", [f], initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        conv = K.conv2d(inputs, self.w, self.strides, padding=self.padding)

        _, h, w, gf = conv.shape.as_list()
        conv = tf.reshape(conv, [-1, h, w, self.n_group, self.n_filter])
        beta = K.zeros([self.n_group, 1])

        for i in range(self.n_iter):
            alpha = K.sigmoid(beta)
            v = K.sum(alpha * conv, axis=-2, keepdims=True)

            if i == self.n_iter - 1:
                return v + self.b

            beta = beta + K.sum(v * conv, axis=[1, 2, -1], keepdims=True)

    def compute_output_shape(self, input_shape):
        _, h, w, gp = input_shape

        return None, h, w, self.n_filter
