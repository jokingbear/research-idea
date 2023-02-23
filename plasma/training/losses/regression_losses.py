import numpy as np

import torch
import torch.nn as nn

from .utils import _assert_inputs


class MSE(nn.Module):

    def forward(self, preds, targets):
        if len(targets.shape) == 1:
            targets = targets[:, np.newaxis]
        
        _assert_inputs(preds, targets)

        return (preds - targets).pow(2).mean()


class PearsonLoss(nn.Module):

    def forward(self, preds, targets):
        if len(targets.shape) == 1:
            targets = targets[:, np.newaxis]
        
        _assert_inputs(preds, targets)
        nclass = targets.shape[1]

        loss = 0
        for i in range(nclass):
            data = torch.stack([preds[:, i], targets[:, i]], dim=0)
            corr = torch.cov(data)[0, 1]
            loss += 1 - corr
        
        return loss
