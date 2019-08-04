from keras import backend as K


def dice_coefficient(y_true, y_pred):
    smooth = K.epsilon()
    
    p = K.sum(y_true * y_pred, axis=(1, 2)) + smooth
    s = K.sum(y_true + y_pred, axis=(1, 2)) + smooth

    dice = K.mean(2 * p / s, axis=-1)

    return K.mean(dice)


def dice_coefficient_int(y_true, y_pred):
    smooth = K.epsilon()
    n_class = int(y_pred.shape[-1])
    axis = K.int_shape(y_pred)[1:-1]

    y_pred = K.argmax(y_pred)
    y_pred = K.one_hot(y_pred, num_classes=n_class)

    p = K.sum(y_true * y_pred, axis=axis)
    s = K.sum(y_true + y_pred, axis=axis)

    dice = (2 * p + smooth) / (s + smooth)

    return K.mean(dice)
