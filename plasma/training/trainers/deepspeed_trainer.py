import deepspeed as ds
import torch.nn as nn

from .standard_trainer import StandardTrainer
from ...functional import auto_func


class ModelEngine(nn.Module):

    def __init__(self, model, loss):
        super().__init__()

        self.model = model
        self.loss = loss
        self.forward_func = auto_func(self.model)
    
    def forward(self, inputs, targets):
        preds = self.forward_func(inputs)

        return preds, self.loss(preds, targets)


class DeepspeedTrainer(StandardTrainer):

    def __init__(self, model, optimizer, loss, metrics=None, dtype='float', rank=0):
        super().__init__(model, optimizer, loss, metrics, dtype, rank)

        model_engine = ModelEngine(model, loss)
        self.engine, _, _, _ = ds.initialize(model=model_engine, optimizer=optimizer)
    
    def _train_one_batch(self, data):
        inputs, targets = data
        preds, loss = self.engine(inputs, targets)

        if isinstance(loss, dict):
            loss_dict = {k: float(loss[k]) for k in loss}
            loss = loss["loss"]
        else:
            loss_dict = {"loss": float(loss)}

        self.engine.backward(loss)
        self.engine.step()

        return loss_dict, preds
