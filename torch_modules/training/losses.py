import torch


def focal_loss_fn(gamma=2):

    def focal_loss(y_true, y_pred):
        if y_pred.shape[1] == 1:
            prob = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        else:
            prob = torch.sum(y_true * y_pred, dim=(1,))

        prob.clamp(1E-7, 1 - 1E-7)
        ln = torch.pow(1 - prob, gamma) * torch.log(prob)

        return (-ln).mean()

    return focal_loss


def dice_loss_fn(n_class=2, rank=2, smooth=1):
    spatial_axes = tuple(range(2, 2 + rank))

    def dice_loss(y_true, y_pred):
        if n_class > 2:
            y_true = y_true[:, 1:, ...]
            y_pred = y_pred[:, 1:, ...]

        p = (y_true * y_pred).sum(dim=spatial_axes)
        s = (y_true + y_pred).sum(dim=spatial_axes)

        dice = (2 * p + smooth) / (s + smooth)

        return (1 - dice).mean()

    return dice_loss


def f1_loss_fn(n_class=2, rank=2, smooth=1):
    spatial_axes = tuple(range(2, 2 + rank))

    def f1_loss(y_true, y_pred):
        if n_class > 2:
            y_true = y_true[:, 1:, ...]
            y_pred = y_pred[:, 1:, ...]

        p = (y_true * y_pred).sum(dim=[0, *spatial_axes])
        s = (y_true + y_pred).sum(dim=[0, *spatial_axes])

        f1 = (2 * p + smooth) / (s + smooth)

        return (1 - f1).mean()

    return f1_loss
