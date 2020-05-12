import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


def _assert_inputs(pred, true):
    assert pred.shape == true.shape, f"predition shape {pred.shape} is not the same as label shape {true.shape}"


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
            sm = np.random.uniform(smooth, 1)
            ln0 = weights[..., 0] * (1 - true) * (sm * ln0 + (1 - sm) * ln1)
            ln1 = weights[..., 1] * true * (sm * ln1 + (1 - sm) * ln0)
        else:
            ln0 = weights[..., 0] * (1 - true) * ln0
            ln1 = weights[..., 1] * true * ln1

        ln = ln0 + ln1
        return -ln.mean()

    return wbce


def get_class_balance_weight(counts):
    total = counts.values[0, 0] + counts.values[0, 1]
    beta = 1 - 1 / total

    weights = (1 - beta) / (1 - beta ** counts)
    normalized_weights = weights / weights.value[:, 0, np.newaxis]

    return normalized_weights


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


def contrastive_loss_fn(t=0.1, normalize=True, ep=1e12):
    entropy_loss = nn.CrossEntropyLoss()

    def contrastive_loss(arg1, arg2):
        _assert_inputs(arg1, arg2)

        if normalize:
            arg1 = func.normalize(arg1)
            arg2 = func.normalize(arg2)

        arg = torch.cat([arg1, arg2], dim=0)
        scores = func.linear(arg, arg)
        diag_scores = scores.diag().diag()
        identity = torch.ones(arg1.shape[0] * 2, device=arg1.device).diag()
        normalized_s = scores - diag_scores - ep * identity
        normalized_s = normalized_s / t

        label1 = torch.arange(0, arg1.shape[0], dtype=torch.long, device=arg1.device)
        label0 = arg1.shape[0] + label1
        label = torch.cat([label0, label1], dim=0)

        return entropy_loss(normalized_s, label)

    return contrastive_loss
