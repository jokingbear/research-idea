import torch
import torch.nn as nn
import torch.nn.functional as func

from .utils import _assert_inputs


def contrastive_loss_fn(t=0.1, normalize=True, ep=1e12, return_acc=True):
    entropy_loss = nn.CrossEntropyLoss()

    def contrastive_loss(arg1, arg2):
        _assert_inputs(arg1, arg2)

        if normalize:
            arg1 = func.normalize(arg1)
            arg2 = func.normalize(arg2)

        arg = torch.cat([arg1, arg2], dim=0)
        scores = func.linear(arg, arg)
        diag_scores = scores.diag().diag()
        identity = torch.ones(arg1.shape[0] * 2, device=arg1.device).diag()
        normalized_s = scores - diag_scores - ep * identity
        normalized_s = normalized_s / t

        label1 = torch.arange(0, arg1.shape[0], dtype=torch.long, device=arg1.device)
        label0 = arg1.shape[0] + label1
        label = torch.cat([label0, label1], dim=0)

        loss = entropy_loss(normalized_s, label)

        if return_acc:
            pick = normalized_s.argmax(dim=1)
            acc = (pick == label).float().mean()

            return {"loss": loss, "acc": acc}

        return loss

    return contrastive_loss
