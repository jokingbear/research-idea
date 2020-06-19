from typing import Tuple

import torch

from .base_trainer import BaseTrainer
from .utils import get_batch_tensors, get_dict


class StandardTrainer(BaseTrainer):

    def __init__(self, model, optimizer, loss, metrics=None,
                 x_device=None, x_type=torch.float, y_device=None, y_type=torch.long):
        """
        :param model: torch module
        :param optimizer: torch optimizer
        :param loss: loss function with signature function(preds, trues)
        :param metrics: list of metric functions with signature function(preds, trues)
        :param x_device: device to put inputs
        :param x_type: type to cast inputs
        :param y_device: device to put labels
        :param y_type: type to cast labels
        """
        super().__init__([model], [optimizer], loss, metrics)

        self.x_device = x_device
        self.x_type = x_type
        self.y_device = y_device
        self.y_type = y_type

        self.training = True

    def extract_data(self, batch_data):
        return get_batch_tensors(batch_data, self.x_type, self.x_device, self.y_type, self.y_device)

    def train_one_batch(self, inputs, targets) -> Tuple[dict, object]:
        if isinstance(inputs, dict):
            preds = self.models[0](**inputs)
        elif isinstance(inputs, list):
            preds = self.models[0](*inputs)
        else:
            preds = self.models[0](inputs)

        loss = self.loss(preds, targets)

        if isinstance(loss, dict):
            loss_dict = {k: float(loss[k]) for k in loss}
            loss = loss["loss"]
        else:
            loss_dict = {"loss": float(loss)}

        loss.backward()
        self.optimizers[0].step()

        return loss_dict, preds

    def get_train_measures(self, inputs, targets, loss_dict, cache) -> dict:
        preds = cache

        measures = loss_dict
        for m in self.metrics:
            m_value = m(preds, targets)
            m_dict = get_dict(m_value, name=m.__name__)
            measures.update(m_dict)

        return measures

    def get_eval_cache(self, inputs, targets):
        if isinstance(inputs, dict):
            preds = self.models[0](**inputs)
        elif isinstance(inputs, list):
            preds = self.models[0](*inputs)
        else:
            preds = self.models[0](inputs)

        return preds, targets

    def get_eval_logs(self, eval_caches):
        preds, trues = eval_caches

        loss = self.loss(preds, trues)
        loss_dict = get_dict(loss, prefix="val ", name="loss")

        measures = loss_dict
        for m in self.metrics:
            m_value = m(preds, trues)
            m_dict = get_dict(m_value, prefix="val ", name=m.__name__)
            measures.update(m_dict)

        return measures
