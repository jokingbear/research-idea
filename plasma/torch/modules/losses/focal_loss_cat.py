import torch
import torch.nn as nn


class CategoricalFocalLoss(nn.Module):

    def __init__(self, gamma=2, dim=-1, logits=True, eps=1e-7) -> None:
        super().__init__()

        self.gamma = gamma
        self.dim = dim
        self.logits = logits
        self.eps = eps
    
    def forward(self, predictions:torch.Tensor, groundtruths:torch.Tensor):
        dim = self.dim
        onehots = torch.nn.functional.one_hot(groundtruths, num_classes=predictions.shape[dim])
        onehots = onehots.to(predictions.dtype)

        if onehots.device != predictions.device:
            onehots = onehots.to(predictions.device)
        
        if self.logits:
            predictions = predictions.softmax(dim=dim)

        probs = (onehots * predictions).sum(dim=dim)
        probs = torch.where(probs > 0.5, probs - self.eps, probs + self.eps)
        loss = (1 - probs).pow(self.gamma) * probs.log()
        return loss.mean()
    
    def extra_repr(self) -> str:
        return f'gamma={self.gamma}, dim={self.dim}, logits={self.logits}, eps={self.eps}'
