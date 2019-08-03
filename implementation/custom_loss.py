from keras import backend as K, losses


def focal_loss(gamma=2):

    def focal(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        prob = K.clip(prob, K.epsilon(), 1 - K.epsilon())

        ln = (1 - prob)**gamma * K.log(prob)
        
        return K.mean(-ln)

    return focal


def binary_focal_loss(gamma=2):

    def focal(y_true, y_pred):
        prob = K.sum(y_true * y_pred + (1 - y_true) * (1 - y_pred), axis=-1)
        prob = K.clip(prob, K.epsilon(), 1 - K.epsilon())

        ln = (1 - prob)**gamma * K.log(prob)

        return K.mean(-ln)

    return focal


def binary_dice_loss(axis=(1, 2), smooth=1):

    def dice_coeff(y_true, y_pred):
        p = K.sum(y_true * y_pred, axis=axis)
        s = K.sum(y_true + y_pred, axis=axis)

        dice = (2 * p + smooth) / (s + smooth)

        return K.mean(1 - dice)

    return dice_coeff


def dice_loss(axis=(1, 2), smooth=1):

    def dice_coeff(y_true, y_pred):
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]

        p = K.sum(y_true * y_pred, axis=axis)
        s = K.sum(y_true + y_pred, axis=axis)

        dice = (2 * p + smooth) / (s + smooth)

        return K.mean(1 - dice)

    return dice_coeff


def bce_dice_loss(dice_loss_fn):

    def bce_dice(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) + dice_loss_fn(y_true, y_pred)

    return bce_dice


def ce_dice_loss(dice_loss_fn):

    def ce_dice(y_true, y_pred):
        return losses.categorical_crossentropy(y_true, y_pred) + dice_loss_fn(y_true, y_pred)

    return ce_dice

