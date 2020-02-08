import torch


def focal_loss_fn(gamma=2, binary=False):
    def focal_loss(y_pred, y_true):
        one = torch.ones([], dtype=torch.float, device=y_pred.device)

        if binary:
            prob = y_true * y_pred + (one - y_true) * (one - y_pred)
        else:
            prob = torch.sum(y_true * y_pred, dim=(1,))

        prob = prob.clamp(1E-7, 1 - 1E-7)
        ln = torch.pow(one - prob, gamma) * torch.log(prob)

        return (-ln).mean()

    return focal_loss


def dice_loss_fn(rank=2, smooth=1, binary=False):
    spatial_axes = tuple(range(2, 2 + rank))

    def dice_loss(y_pred, y_true):
        if not binary:
            y_true = y_true[:, 1:, ...]
            y_pred = y_pred[:, 1:, ...]

        p = (y_true * y_pred).sum(dim=spatial_axes)
        s = (y_true + y_pred).sum(dim=spatial_axes)

        dice = (2 * p + smooth) / (s + smooth)

        return (1 - dice).mean()

    return dice_loss


def f1_loss_fn(binary=False, smooth=1):
    def f1_loss(y_pred, y_true):
        if not binary:
            y_true = y_true[:, 1:, ...]
            y_pred = y_pred[:, 1:, ...]

        p = (y_true * y_pred).sum(dim=0)
        s = (y_true + y_pred).sum(dim=0)

        f1 = (2 * p + smooth) / (s + smooth)

        return (1 - f1).mean()

    return f1_loss


def weighted_bce(weights):

    def loss(y_pred, y_true):
        prob = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        prob = prob.clamp(1e-7, 1 - 1e-7)

        weight = weights[1, ...] * y_true + weights[0, ...] * (1 - y_true)
        log = weight * torch.log(prob)

        return (-log).mean()

    return loss


def combine_loss(*losses, weights=None):
    weights = weights or [1] * len(losses)

    def total_loss(y_pred, y_true):
        loss = 0

        for ls, w in zip(losses, weights):
            loss = loss + w * ls(y_pred, y_true)

        return loss

    return total_loss
