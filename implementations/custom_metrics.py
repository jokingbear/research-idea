import tensorflow as tf


def binary_dice_coefficient_metrics(smooth=1, cast=True, threshold=0.5):

    def dice_coefficient(y_true, y_pred):
        if cast:
            y_pred = tf.cast(y_pred >= threshold, tf.float32)

        spatial_shape = y_pred.shape.as_list()[1:-1]
        axis = [i + 1 for i in range(spatial_shape)]

        p = tf.reduce_sum(y_true * y_pred, axis=axis)
        s = tf.reduce_sum(y_true + y_pred, axis=axis)

        dice = (2 * p + smooth) / (s + smooth)

        return tf.reduce_mean(dice)

    return dice_coefficient


def cat_dice_coefficient_metrics(smooth=1, cast=True):

    def dice_coefficient(y_true, y_pred):
        if cast:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred = tf.one_hot(y_pred, axis=-1, dtype=tf.float32)

        y_pred = y_pred[..., 1:]
        y_true = y_true[..., 1:]

        spatial_shape = y_pred.shape.as_list()[1:-1]
        axis = [i + 1 for i in range(len(spatial_shape))]

        p = tf.reduce_sum(y_true * y_pred, axis=axis)
        s = tf.reduce_sum(y_true + y_pred, axis=axis)

        dice = (2 * p + smooth) / (s + smooth)

        return tf.reduce_mean(dice)

    return dice_coefficient


def f1_score(y_true, y_pred, smooth=1):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    p = tf.reduce_sum(y_true * y_pred)
    s = tf.reduce_sum(y_true + y_pred)

    f1 = (2 * p + smooth) / (s + smooth)

    return f1
