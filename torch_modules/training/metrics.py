import torch


def accuracy(y_true, y_pred):
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0, ...]
        y_pred = y_pred >= 0.5
    elif y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    acc = (y_true == y_pred).type(torch.FloatTensor)

    return torch.mean(acc)
