import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .utils import _assert_inputs


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, binary=False):
        super().__init__()

        self.gamma = gamma
        self.binary = binary

    def forward(self, preds, trues):
        _assert_inputs(preds, trues)

        if self.binary:
            prob = trues * preds + (1 - trues) * (1 - preds)
        else:
            prob = (trues * preds).sum(dim=1)

        ln = (1 - prob).pow(self.gamma) * (prob + 1e-7).log()

        return -ln.mean()

    def extra_repr(self):
        return f"gamma={self.gamma}, binary={self.binary}"


class FbetaLoss(nn.Module):
    
    def __init__(self, beta=1, axes=(0,), binary=False, smooth=1e-7):
        super().__init__()

        self.beta = beta
        self.axes = axes
        self.binary = binary
        self.smooth = smooth

    def forward(self, preds, trues):
        beta2 = self.beta ** 2

        if not self.binary:
            trues = trues[:, 1:, ...]
            preds = preds[:, 1:, ...]

        _assert_inputs(preds, trues)

        p = (beta2 + 1) * (trues * preds).sum(dim=self.axes)
        s = (beta2 * trues + preds).sum(dim=self.axes)

        fb = (p + self.smooth) / (s + self.smooth)

        return (1 - fb).mean()

    def extra_repr(self):
        return f"beta={self.beta}, axes={self.axes}, binary={self.binary}, smooth={self.smooth}"


class WBCE(nn.Module):

    def __init__(self, weights_path=None, smooth=None):
        super().__init__()

        if weights_path is None:
            weights = np.array([[1, 1]])
            print('using default weight')
            print(pd.DataFrame(weights, index=['default']))
        elif ".csv" in weights_path:
            weights = pd.read_csv(weights_path, index_col=0)
            print(weights)
            weights = weights.values
        elif ".npy" in weights_path:
            weights = np.load(weights_path)
            print(weights.shape)
        else:
            raise NotImplementedError("only support csv and numpy extension")

        self.weights = torch.tensor(weights, dtype=torch.float)
        self.smooth = smooth

    def forward(self, preds, trues):
        _assert_inputs(preds, trues)

        ln0 = (1 - preds + 1e-7).log()
        ln1 = (preds + 1e-7).log()

        weights = self.weights.to(preds.device)
        if self.smooth is not None:
            sm = torch.ones_like(preds).uniform_(1 - self.smooth, 1)
            ln0 = weights[..., 0] * (1 - trues) * (sm * ln0 + (1 - sm) * ln1)
            ln1 = weights[..., 1] * trues * (sm * ln1 + (1 - sm) * ln0)
        else:
            ln0 = weights[..., 0] * (1 - trues) * ln0
            ln1 = weights[..., 1] * trues * ln1

        ln = ln0 + ln1
        return -ln.mean()

    def extra_repr(self):
        return f"weights_shape={self.weights.shape}, smooth={self.smooth}, sym={self.sym}"

    @staticmethod
    def get_class_balance_weight(counts, anchor=0):
        """
        calculate class balance weight from counts with anchor
        :param counts: class counts, shape=(n_class, 2)
        :param anchor: make anchor class weight = 1 and keep the aspect ratio of other weight
        :return: weights for cross entropy loss
        """
        total = counts.values[0, 0] + counts.values[0, 1]
        beta = 1 - 1 / total

        weights = (1 - beta) / (1 - beta ** counts)
        normalized_weights = weights / weights.values[:, anchor, np.newaxis]

        return normalized_weights


class CrossEntropy(nn.Module):

    def __init__(self, binary=False, logit=False, smooth=None, eps=1e-7):
        super().__init__()

        self.binary = binary
        self.logit = logit
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds, targets):
        if not self.binary and len(targets.shape) <= 2:
            targets = targets[:, 0] if len(targets.shape) == 2 else targets

            if targets.dtype != torch.long:
                targets = targets.type(torch.long)

            targets = nn.functional.one_hot(targets, num_classes=preds.shape[-1])
        
        if self.smooth is not None:
            if self.binary: 
                smooth = torch.ones_like(targets).uniform_(self.smooth, 1)
                targets = targets * smooth + (1 - targets) * (1 - smooth)
            else:
                smooth = torch.ones_like(targets[:, 0]).uniform_(self.smooth, 1)
                smooth = smooth.expand(-1, targets.shape[-1])
                targets = targets * smooth + (1 - targets) * (1 - smooth) / (targets.shape[-1] - 1) 

        if self.logit:
            if self.binary:
                preds = torch.sigmoid(preds)
            else:
                preds = torch.softmax(preds, dim=1)

        _assert_inputs(preds, targets)
        
        if self.binary:
            cross_entropy = -targets * (preds + self.eps).log() - (1 - targets) * (1 - preds + self.eps).log()
        else:
            cross_entropy = (-targets * (preds + self.eps).log()).sum(dim=1)
        
        return cross_entropy.mean()
    
    def extra_repr(self) -> str:
        return f'binary={self.binary}, logit={self.logit}, smooth={self.smooth}, eps={self.eps}'
