import torch


def accuracy(y_true, y_pred):
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0, ...]
        y_pred = y_pred >= 0.5
    elif y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    acc = (y_true == y_pred).float()

    return torch.mean(acc)


def dice_coefficient_fn(n_class=2, rank=2, smooth=1E-7):
    spatial_axes = tuple(range(2, 2 + rank))

    def dice_coefficient(y_true, y_pred):
        if n_class > 2:
            y_true = y_true[:, 1:, ...]
            y_pred = y_pred[:, 1:, ...]

        p = (y_true * y_pred).sum(dim=spatial_axes)
        s = (y_true + y_pred).sum(dim=spatial_axes)

        dice = (2 * p + smooth) / (s + smooth)

        return dice.mean()

    return dice_coefficient
