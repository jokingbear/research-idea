import torch
import torch.nn.functional as func


def accuracy(y_pred, y_true):
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0, ...]
        y_pred = y_pred >= 0.5
    elif y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    acc = (y_true == y_pred).float()

    return torch.mean(acc)


def dice_coefficient_fn(n_class=2, input_rank=4, smooth=1e-7, binary=False, cast=False):
    assert input_rank >= 3, "input_rank must be bigger than 2"

    spatial_axes = list(range(2, input_rank))

    def dice_coefficient(y_pred, y_true):
        if binary and cast:
            y_pred = (y_pred >= 0.5).type(torch.float)
        elif cast:
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = func.one_hot(y_pred, num_classes=n_class).type(torch.float)
            y_pred = y_pred.permute(0, -1, *[a - 1 for a in spatial_axes])

        y_pred = y_pred[:, 1:, ...] if not binary else y_pred
        y_true = y_true[:, 1:, ...] if not binary else y_true

        p = (y_true * y_pred).sum(dim=spatial_axes)
        s = (y_true + y_pred).sum(dim=spatial_axes)

        dice = (2 * p + smooth) / (s + smooth)

        return dice.mean()

    return dice_coefficient


def f1_fn(n_class=2, smooth=1e-7, binary=False, cast=False):

    def f1_score(y_pred, y_true):
        if binary and cast:
            y_pred = (y_pred >= 0.5).type(torch.float)
        elif cast:
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = func.one_hot(y_pred, num_classes=n_class).type(torch.float)

        p = (y_true * y_pred).sum(dim=0)
        s = (y_true + y_pred).sum(dim=0)

        f1 = (2 * p + smooth) / (s + smooth)

        return f1.mean()

    return f1_score
