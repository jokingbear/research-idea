import numpy as np

from plasma.training.callbacks.base_class import Callback


class CutMix(Callback):

    def __init__(self, alpha=1):
        super().__init__()

        self.alpha = alpha

    def on_training_batch_begin(self, batch, x, y):
        shuffled_idc = np.random.choice(x.shape[0], size=x.shape[0], replace=True)
        shuffled_x = x[shuffled_idc]
        shuffled_y = y[shuffled_idc]
        lmbda = np.random.beta(self.alpha, self.alpha)

        h, w = x.shape[-2:]
        r_x = np.random.randint(0, w)
        r_y = np.random.randint(0, h)
        r_w = np.sqrt(1 - lmbda)
        r_h = np.sqrt(1 - lmbda)

        x1 = np.round(np.clip(r_x - r_w / 2, a_min=0))
        x2 = np.round(np.clip(r_x + r_w / 2, a_max=w))
        y1 = np.round(np.clip(r_y - r_h / 2, a_min=0))
        y2 = np.round(np.clip(r_y + r_h / 2, a_max=h))

        lmbda = 1 - (x2 - x1) * (y2 - y1) / h / w
        x[..., y1:y2, x1:x2] = shuffled_x[..., y1:y2, x1:x2]
        y[...] = lmbda * y + (1 - lmbda) * shuffled_y
