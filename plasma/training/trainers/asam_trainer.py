from typing import Tuple
from .standard_trainer import StandardTrainer

import torch


class ASAM(StandardTrainer):

    def __init__(self, model, optimizer, loss, metrics=None, dtype='float', rank=0, rho=0.5, eta=0):
        super().__init__(model, optimizer, loss, metrics, dtype, rank)

        self.rho = rho
        self.eta = eta

    def _train_one_batch(self, data) -> Tuple[dict, object]:
        inputs, targets = data
        if isinstance(inputs, dict):
            preds = self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            preds = self.model(*inputs)
        else:
            preds = self.model(inputs)
        
        loss = self.loss(preds, targets)

        if isinstance(loss, dict):
            loss_dict = {k: float(loss[k]) for k in loss}
            loss = loss["loss"]
        else:
            loss_dict = {"loss": float(loss)}
        
        loss.backward()
        self.ascent_step()

        loss = self.loss(preds, targets)
        if isinstance(loss, dict):
            loss = loss["loss"]

        loss.backward()
        self.descent_step()

        return loss_dict

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

    def extra_repr(self):
        return f'rho={self.rho}, eta={self.eta}'