def focal_loss_fn(gamma=2, binary=False):
    def focal_loss(pred, true):
        if binary:
            prob = true * pred + (1 - true) * (1 - pred)
        else:
            prob = (true * pred).sum(dim=1)

        ln = (1 - prob).pow(gamma) * (prob + 1e-7).log()

        return -ln.mean()

    return focal_loss


def dice_loss_fn(rank=2, smooth=1e-7, binary=False):
    axes = tuple(range(2, 2 + rank))

    def dice_loss(pred, true):
        if not binary:
            true = true[:, 1:, ...]
            pred = pred[:, 1:, ...]

        p = 2 * (true * pred).sum(dim=axes)
        s = (true + pred).sum(dim=axes)

        dice = (p + smooth) / (s + smooth)

        return (1 - dice).mean()

    return dice_loss


def fb_loss_fn(beta=1, binary=False, smooth=1e-7):
    beta2 = beta ** 2

    def fb_loss(pred, true):
        if not binary:
            true = true[:, 1:, ...]
            pred = pred[:, 1:, ...]

        p = (beta2 + 1) * (true * pred).sum(dim=0)
        s = (beta2 * true + pred).sum(dim=0)

        fb = (p + smooth) / (s + smooth)

        return (1 - fb).mean()

    return fb_loss


def weighted_bce(weights):

    def loss(pred, true):
        ln0 = weights[..., 0] * (1 - true) * (1 - pred + 1e-7).log()
        ln1 = weights[..., 1] * true * (pred + 1e-7).log()

        ln = ln0 + ln1
        return -ln.mean()

    return loss


def combine_loss(*losses, weights=None):
    weights = weights or [1] * len(losses)

    def total_loss(pred, true):
        loss = 0

        for ls, w in zip(losses, weights):
            loss = loss + w * ls(pred, true)

        return loss

    return total_loss
