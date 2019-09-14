import torch


def focal_loss_fn(gamma=2):

    def focal_loss(y_true, y_pred):
        if y_pred.shape[1] == 1:
            prob = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            prob = prob.clamp(1E-7, 1 - 1E-7)
        else:
            prob = torch.sum(y_true * y_pred, dim=(1,))

        ln = torch.pow(1 - prob, gamma) * torch.log(prob)

        return torch.mean(ln, dim=(0,))

    return focal_loss


def dice_loss_fn(n_class=2, rank=2, smooth=1E-7):
    spatial_axes = tuple([2 + i for i in range(rank)])

    def dice_loss(y_true, y_pred):
        p = torch.sum(y_true * y_pred, dim=spatial_axes)
        s = torch.sum(y_true + y_pred, dim=spatial_axes)

        if n_class > 2:
            p = p[:, 1:, ...]
            s = s[:, 1:, ...]

        dice = (2 * p + smooth) / (s + smooth)

        return torch.mean(1 - dice)

    return dice_loss


def f1_loss_fn(n_class=2, rank=2, smooth=1E-7):
    spatial_axes = tuple([2 + i for i in range(rank)])

    def f1_loss(y_true, y_pred):
        p = torch.sum(y_true * y_pred, dim=(0, *spatial_axes))
        s = torch.sum(y_true + y_pred, dim=(0, *spatial_axes))

        if n_class > 2:
            p = p[:, 1:, ...]
            s = s[:, 1:, ...]

        f1 = (2 * p + smooth) / (s + smooth)

        return torch.mean(1 - f1)

    return f1_loss