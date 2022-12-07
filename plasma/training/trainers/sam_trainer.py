from typing import Tuple
from .standard_trainer import StandardTrainer
from collections import defaultdict

import torch


class SAM(StandardTrainer):

    def __init__(self, model, optimizer, loss, metrics=None, dtype='float', rank=0, rho=0.5, adaptive=True):
        super().__init__(model, optimizer, loss, metrics, dtype, rank)

        self.minimizer = SAMOptimizer(self.model.parameters(), self.optimizer, rho, adaptive)
        self.rho = rho
        self.adaptive = adaptive
        self.state = defaultdict(dict)

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
        self.minimizer.first_step(zero_grad=True)

        final_preds = preds
        if isinstance(inputs, dict):
            preds = self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            preds = self.model(*inputs)
        else:
            preds = self.model(inputs)

        loss = self.loss(preds, targets)
        if isinstance(loss, dict):
            loss = loss["loss"]

        loss.backward()
        self.minimizer.second_step(zero_grad=True)

        return loss_dict, final_preds

    def extra_repr(self):
        return f'rho={self.rho}, adaptive={self.adaptive}'


class SAMOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.defaults["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.defaults["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.defaults['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
