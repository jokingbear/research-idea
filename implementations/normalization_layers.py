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
        shape = inputs.shape.as_list()[1:]
        spatial_shape = shape[:-1]
        gp = shape[-1]
        g = self.n_groups

        x = K.reshape(inputs, (-1,) + spatial_shape + (g, gp // g))

        mean, var = tf.nn.moments(x, spatial_shape + (-1, ), keep_dims=True)

        x = (x - mean) / (tf.sqrt(var) + 1E-7)

        x = tf.reshape(x, shape)

        return x * self.gamma + self.beta

