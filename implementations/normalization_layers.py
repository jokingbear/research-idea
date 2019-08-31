import tensorflow as tf

from tensorflow.keras import layers, initializers as inits


class GroupNorm(layers.Layer):

    def __init__(self, n_group, gamma_initializer="ones", **kwargs):
        self.n_group = n_group
        self.gamma_initializer = inits.get(gamma_initializer)

        super().__init__(**kwargs)

        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        gp = input_shape[-1]
        
        self.gamma = self.add_weight("gamma", [gp], initializer=self.gamma_initializer)
        self.beta = self.add_weight("beta", [gp], initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        shape = inputs.shape
        spatial_shape = shape[1:-1]
        gp = shape[-1]
        g = self.n_group
        spatial_axis = [i + 1 for i in range(len(spatial_shape))]
        x = tf.reshape(inputs, (-1, *spatial_shape, g, gp // g))

        mean, var = tf.nn.moments(x, spatial_axis + [-1], keepdims=True)

        x = (x - mean) / (tf.sqrt(var) + 1E-7)

        x = tf.reshape(x, (-1, *shape[1:]))

        return x * self.gamma + self.beta

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()

        config["n_group"] = self.n_group
        config["gamma_initializer"] = inits.serialize(self.gamma_initializer)

        return config


class InstanceNorm(layers.Layer):

    def __init__(self, gamma_initializer="ones", **kwargs):
        super().__init__(**kwargs)

        self.gamma_initializer = inits.get(gamma_initializer)
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        f = input_shape[-1]
        self.gamma = self.add_weight("gamma", shape=[f], initializer=self.gamma_initializer)
        self.beta = self.add_weight("beta", shape=[f], initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        spatial = inputs.shape[1:-1]
        spatial_axes = [i + 1 for i in range(len(spatial))]
        mean, var = tf.nn.moments(inputs, spatial_axes, keepdims=True)

        std = tf.sqrt(var) + 1E-7

        x = (inputs - mean) / std
        x = self.gamma * x + self.beta

        return x

    def compute_output_signature(self, input_signature):
        return input_signature

    def get_config(self):
        config = super().get_config()

        config["gamma_initializer"] = inits.serialize(self.gamma_initializer)

        return config
