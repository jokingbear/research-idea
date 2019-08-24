import tensorflow as tf

from tensorflow.python.keras import losses


def focal_loss(gamma=2):
    def focal(y_true, y_pred):
        prob = tf.reduce_sum(y_true * y_pred, axis=-1)
        prob = tf.clip_by_value(prob, 1E-7, 1 - 1E-7)

        ln = tf.pow(1 - prob, gamma) * tf.math.log(prob)

        return tf.reduce_mean(-ln)

    return focal


def binary_focal_loss(gamma=2):
    def focal(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        prob = tf.reduce_sum(y_true * y_pred + (1 - y_true) * (1 - y_pred))
        prob = tf.clip_by_value(prob, 1E-7, 1 - 1E-7)

        ln = (1 - prob) ** gamma * tf.math.log(prob)

        return tf.reduce_mean(-ln)

    return focal


def dice_loss(categorical=True, smooth=1, include_batch=False):
    def dice_coeff(y_true, y_pred):
        if categorical:
            y_true = y_true[..., 1:]
            y_pred = y_pred[..., 1:]

        spatial_axes = list(range(len(y_pred.shape)))[1:-1]
        axes = [0] + spatial_axes if include_batch else spatial_axes

        p = tf.reduce_sum(y_true * y_pred, axis=axes)
        s = tf.reduce_sum(y_true + y_pred, axis=axes)

        dice = (2 * p + smooth) / (s + smooth)

        return tf.reduce_mean(1 - dice)

    return dice_coeff


def combine_loss(losses, weights=None):
    weights = weights if weights else [1] * len(losses)

    def loss(y_true, y_pred):
        final = losses[0](y_true, y_pred) * weights[0]

        for l, w in zip(losses[1:], weights[1:]):
            final = final + l(y_true, y_pred) * w

        return final

    return loss