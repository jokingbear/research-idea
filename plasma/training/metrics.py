import torch


def dice_coefficient_fn(input_rank=4, smooth=1e-7, binary=False, cast=False):
    assert input_rank >= 3, "input_rank must be bigger than 2"

    axes = list(range(2, input_rank))

    def dice_coefficient(pred, true):
        if cast:
            if binary:
                pred = (pred >= 0.5).type(torch.float)
            else:
                n_class = pred.shape[1]
                pred = pred.argmax(dim=1)
                pred = torch.stack([pred == i for i in range(1, n_class)], dim=1)

        true = true[:, 1:, ...] if not binary else true

        p = 2 * (true * pred).sum(dim=axes)
        s = (true + pred).sum(dim=axes)

        dice = (p + smooth) / (s + smooth)

        return dice.mean()

    return dice_coefficient


def fb_fn(beta=1, smooth=1e-7, binary=False, cast=False):
    beta2 = beta ** 2

    def fb_score(pred, true):
        if binary and cast:
            pred = (pred >= 0.5).type(torch.float)
        elif cast:
            n_class = pred.shape[1]
            pred = torch.argmax(pred, dim=1)
            pred = torch.stack([pred == c for c in range(1, n_class)], dim=1)

        true = true[:, 1:, ...] if not binary else true

        p = (beta2 + 1) * (true * pred).sum(dim=0)
        s = (beta2 * true + pred).sum(dim=0)

        fb = (p + smooth) / (s + smooth)

        return fb.mean()

    return fb_score


def acc_fn(binary=False):
    def acc(pred, true):
        if binary:
            pred = (pred >= 0.5)
        else:
            pred = pred.argmax(dim=1)

        result = (pred == true).float().mean()
        return result

    return acc
