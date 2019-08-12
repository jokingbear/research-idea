import tensorflow as tf

from keras import backend as K, layers, initializers


class GroupNorm(layers.Layer):

    def __init__(self, n_group, gamma_initializer="ones", **kwargs):
        self.n_group = n_group
        self.gamma_initializer = initializers.get(gamma_initializer)

        super().__init__(**kwargs)

        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        gp = input_shape[-1]
        
        self.gamma = self.add_weight("gamma", [gp], initializer=self.gamma_initializer)
        self.beta = self.add_weight("beta", [gp], initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        shape = inputs.shape.as_list()
        spatial_shape = shape[1:-1]
        gp = shape[-1]
        g = self.n_group
        spatial_axis = [i + 1 for i in range(len(spatial_shape))]
        x = K.reshape(inputs, [-1] + spatial_shape + [g, gp // g])

        mean, var = tf.nn.moments(x, spatial_axis + [-1], keep_dims=True)

        x = (x - mean) / (tf.sqrt(var) + 1E-7)

        x = tf.reshape(x, [-1] + shape[1:])

        return x * self.gamma + self.beta
