from keras import backend as K


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
