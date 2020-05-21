import numpy as np
import torch

from .utils import _assert_inputs


def focal_loss_fn(gamma=2, binary=False, one_hot_n_class=None):
    def focal_loss(pred, true):
        if one_hot_n_class is not None:
            true = torch.stack([true == i for i in range(one_hot_n_class)], dim=1)

        _assert_inputs(pred, true)

        if binary:
            prob = true * pred + (1 - true) * (1 - pred)
        else:
            prob = (true * pred).sum(dim=1)

        ln = (1 - prob).pow(gamma) * (prob + 1e-7).log()

        return -ln.mean()

    return focal_loss


def fb_loss_fn(beta=1, axes=(0,), binary=False, one_hot_n_class=None, smooth=1e-7):
    beta2 = beta ** 2

    def fb_loss(pred, true):
        if one_hot_n_class is not None:
            true = torch.stack([true == i for i in range(one_hot_n_class)], dim=1)

        if not binary:
            true = true[:, 1:, ...]
            pred = pred[:, 1:, ...]

        _assert_inputs(pred, true)

        p = (beta2 + 1) * (true * pred).sum(dim=axes)
        s = (beta2 * true + pred).sum(dim=axes)

        fb = (p + smooth) / (s + smooth)

        return (1 - fb).mean()

    return fb_loss


def weighted_bce(weights, smooth=None):
    def wbce(pred, true):
        _assert_inputs(pred, true)

        ln0 = (1 - pred + 1e-7).log()
        ln1 = (pred + 1e-7).log()

        if smooth is not None:
            sm = np.random.uniform(1 - smooth, 1)
            ln0 = weights[..., 0] * (1 - true) * (sm * ln0 + (1 - sm) * ln1)
            ln1 = weights[..., 1] * true * (sm * ln1 + (1 - sm) * ln0)
        else:
            ln0 = weights[..., 0] * (1 - true) * ln0
            ln1 = weights[..., 1] * true * ln1

        ln = ln0 + ln1
        return -ln.mean()

    return wbce


def combine_loss(*losses, weights=None):
    weights = weights or [1] * len(losses)

    def total_loss(pred, true):
        loss = 0
        d = {}

        for ls, w in zip(losses, weights):
            ind_loss = ls(pred, true)
            d[ls.__name__] = ind_loss
            loss = loss + w * ind_loss

        d["loss"] = loss
        return d

    return total_loss
