import tensorflow as tf

from keras import backend as K, layers, initializers


class GroupNorm(layers.Layer):

    def __init__(self, n_groups,
                 gamma_initializer="ones", **kwargs):
        self.n_groups = n_groups
        self.gamma_initializer = initializers.get(gamma_initializer)

        super().__init__(**kwargs)

    def build(self, input_shape):
        gp = input_shape[-1]
        
        self.gamma = self.add_weight("gamma", [gp], initializer=self.gamma_initializer)
        self.beta = self.add_weight("beta", [gp], initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs (?, h, w, cp)
        h, w, gp = K.int_shape(inputs)[1:]
        g = self.n_groups

        # (?, h, w, c, p)
        x = K.reshape(inputs, [-1, h, w, g, gp // g])

        # (?, 1, 1, c, 1)
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)

        # (?, 1, 1, c, 1)
        x = (x - mean) / (K.sqrt(var) + K.epsilon())

        # (?, 1, 1, c, 1)
        x = K.reshape(x, [-1, h, w, gp])

        return x * self.gamma + self.beta

