import tensorflow as tf


def f1_score(y_true, y_pred, smooth=1):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    p = tf.reduce_sum(y_true * y_pred)
    s = tf.reduce_sum(y_true + y_pred)

    f1 = (2 * p + smooth) / (s + smooth)

    return f1


def dice_coefficient_metrics(n_class, cast=True, threshold=0.5, smooth=1):
    def dice_coefficient(y_true, y_pred):
        if y_true.shape != y_pred.shape:
            if n_class == 2:
                y_true = y_true[..., tf.newaxis]
            else:
                y_true = tf.one_hot(y_true, depth=n_class)

        if cast:
            if n_class == 2:
                y_pred = tf.cast(y_pred >= threshold, tf.float32)
            else:
                y_pred = tf.argmax(y_pred, axis=-1)
                y_pred = tf.one_hot(y_pred, depth=n_class)

        spatial_axes = list(range(len(y_pred.shape)))[1:-1]
        p = tf.reduce_sum(y_true * y_pred, axis=spatial_axes)
        s = tf.reduce_sum(y_true + y_pred, axis=spatial_axes)

        dice = (2 * p + smooth) / (s + smooth)
        return tf.reduce_mean(dice)

    return dice_coefficient
