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
