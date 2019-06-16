from keras import backend as K


def dice_coefficient_metrics(smooth=K.epsilon(), ignore_bg=False):

    def dice_coefficient(y_true, y_pred):
        y_true = y_true[..., 1:] if ignore_bg else y_true
        y_pred = y_pred[..., 1:] if ignore_bg else y_pred
        
        p = K.sum(y_true * y_pred, axis=(1, 2)) + smooth
        s = K.sum(y_true + y_pred, axis=(1, 2)) + smooth

        dice = K.mean(2 * p / s, axis=-1)

        return K.mean(dice)

    return dice_coefficient


def jaccard_coefficient_metrics(smooth=K.epsilon(), ignore_bg=False):

    def jaccard_coefficient(y_true, y_pred):
        y_true = y_true[..., 1:] if ignore_bg else y_true
        y_pred = y_pred[..., 1:] if ignore_bg else y_pred

        p = K.sum(y_true * y_pred, axis=(1, 2))
        s = K.sum(y_true + y_pred, axis=(1, 2))

        j = K.mean((p + smooth) / (s - p + smooth), axis=-1)

        return K.mean(j)

    return jaccard_coefficient
