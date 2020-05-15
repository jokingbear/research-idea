import torch
import torch.nn.functional as func


def contrastive_acc_fn(t=0.1, normalize=True, ep=1e12):

    def contrastive_acc(arg1, arg2):
        if normalize:
            arg1 = func.normalize(arg1)
            arg2 = func.normalize(arg2)

        arg = torch.cat([arg1, arg2], dim=0)
        scores = func.linear(arg, arg)
        diag_scores = scores.diag().diag()
        identity = torch.ones(arg1.shape[0] * 2, device=arg1.device).diag()
        normalized_s = scores - diag_scores - ep * identity
        normalized_s = normalized_s / t
        pick = normalized_s.argmax(dim=-1)

        label1 = torch.arange(0, arg1.shape[0], dtype=torch.long, device=arg1.device)
        label0 = arg1.shape[0] + label1
        label = torch.cat([label0, label1], dim=0)

        return (pick == label).float().mean()

    return contrastive_acc
