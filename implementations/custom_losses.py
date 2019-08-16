import tensorflow as tf

losses = tf.keras.losses


def focal_loss(gamma=2):
    def focal(y_true, y_pred):
        prob = tf.reduce_sum(y_true * y_pred, axis=-1)
        prob = tf.clip_by_value(prob, 1E-7, 1 - 1E-7)

        ln = (1 - prob) ** gamma * tf.math.log(prob)

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


def dice_loss(smooth=1, categorical=True):
    def dice_coeff(y_true, y_pred):
        shape = y_pred.shape.as_list()[1:-1]

        if categorical:
            y_true = y_true[..., 1:]
            y_pred = y_pred[..., 1:]

        p = tf.reduce_sum(y_true * y_pred, axis=shape)
        s = tf.reduce_sum(y_true + y_pred, axis=shape)

        dice = (2 * p + smooth) / (s + smooth)

        return tf.reduce_mean(1 - dice)

    return dice_coeff


def bce_dice(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(categorical=False)(y_true, y_pred)


def ce_dice(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred) + dice_loss()(y_true, y_pred)
