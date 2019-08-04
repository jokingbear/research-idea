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
        inputs = K.reshape(inputs, shape[:-1] + (g, c))

        beta = K.zeros([1, 1, 1, g, 1])
        i0 = K.constant(0)
        v, _ = tf.while_loop(lambda i, *args: i < self.n_iter, self._routing, [i0, inputs, beta],
                             shape_invariants=[tf.TensorShape(0), tf.TensorShape([None] + shape[1:-1] + [1, c]),
                                               tf.TensorShape([None, 1, 1, g, 1])], swap_memory=True)
        
        return v[..., 0, :]

    def _routing(self, i, inputs, beta):
        alpha = K.sigmoid(beta)
        v = K.sum(alpha * inputs, axis=-2, keepdims=True)

        return tf.cond(i < self.n_iter - 1, lambda: (v, beta + K.sum(v * inputs, axis=(1, 2, -1))),
                       lambda: (v, beta))
