import torch


def fb_fn(beta=1, axes=(0,), smooth=1e-7, binary=False, one_hot_n_class=None, cast=False):
    beta2 = beta ** 2

    def fb_score(pred, true):
        if binary and cast:
            pred = (pred >= 0.5).type(torch.float)
        elif cast:
            n_class = pred.shape[1]
            pred = torch.argmax(pred, dim=1)
            pred = torch.stack([pred == c for c in range(1, n_class)], dim=1)

        if one_hot_n_class is not None:
            true = torch.stack([true == i for i in range(one_hot_n_class)], dim=1)

        true = true[:, 1:, ...] if not binary else true

        p = (beta2 + 1) * (true * pred).sum(dim=axes)
        s = (beta2 * true + pred).sum(dim=axes)

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
