def acc_fn(binary=False):
    def acc(pred, true):
        if binary:
            pred = (pred >= 0.5)
        else:
            pred = pred.argmax(dim=1)

        result = (pred == true).float().mean()
        return result

    return acc


def fb_loss_fn(beta=1, axes=(0,), binary=False, smooth=1e-7):
    beta2 = beta ** 2

    def fb_score(pred, true):
        if not binary:
            true = true[:, 1:, ...]
            pred = pred[:, 1:, ...]

        p = (beta2 + 1) * (true * pred).sum(dim=axes)
        s = (beta2 * true + pred).sum(dim=axes)

        fb = (p + smooth) / (s + smooth)

        return fb.mean()

    return fb_score
