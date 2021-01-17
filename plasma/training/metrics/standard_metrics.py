import torch
import torch.nn as nn
import numpy as np


class Accuracy(nn.Module):

    def __init__(self, binary=False):
        super().__init__()

        self.binary = binary

    def forward(self, preds, trues):
        if self.binary:
            preds = (preds >= 0.5)
        else:
            preds = preds.argmax(dim=1)

        result = (preds == trues).float().mean()
        return result

    def extra_repr(self):
        return f"binary={self.binary}"


class FbetaScore(nn.Module):

    def __init__(self, beta=1, axes=(0,), binary=False, smooth=1e-7, classes=None):
        super().__init__()

        self.beta = beta
        self.axes = axes
        self.binary = binary
        self.smooth = smooth
        self.classes = classes

    def forward(self, preds, trues):
        beta2 = self.beta ** 2

        if self.binary:
            preds = (preds >= 0.5).float()
        else:
            trues = trues[:, 1:, ...]
            preds = preds.argmax(dim=1)
            preds = torch.stack([(preds == i).float() for i in range(1, trues.shape[1])], dim=1)

        p = (beta2 + 1) * (trues * preds).sum(dim=self.axes)
        s = (beta2 * trues + preds).sum(dim=self.axes)

        fb = (p + self.smooth) / (s + self.smooth)

        if self.classes is not None:
            if len(fb.shape) == 1:
                fb = fb[np.newaxis]
            results = {f"{c}_F{self.beta}": fb[:, i].mean() for i, c in enumerate(self.classes)}
            results[f"F{self.beta}"] = fb.mean()
            return results

        return {f"F{self.beta}": fb.mean()}

    def extra_repr(self):
        return f"beta={self.beta}, axes={self.axes}, binary={self.binary}, " \
               f"smooth={self.smooth}, classes={self.classes}"
