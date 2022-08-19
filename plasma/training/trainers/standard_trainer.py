from typing import Tuple

import torch

from .base_trainer import BaseTrainer
from .utils import get_dict


class StandardTrainer(BaseTrainer):

    def __init__(self, model, optimizer, loss, metrics=None, dtype='float', device='cuda:0'):
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
        super().__init__([model], [optimizer], loss, metrics, dtype, device)

        self.training = True

    def _train_one_batch(self, data) -> Tuple[dict, object]:
        inputs, targets = data
        if isinstance(inputs, dict):
            preds = self.models[0](**inputs)
        elif isinstance(inputs, (list, tuple)):
            preds = self.models[0](*inputs)
        else:
            preds = self.models[0](inputs)

        loss = self.loss(preds, targets)

        if isinstance(loss, dict):
            loss_dict = {k: float(loss[k]) for k in loss}
            loss = loss["Loss"]
        else:
            loss_dict = {"Loss": float(loss)}

        loss.backward()
        self.optimizers[0].step()

        return loss_dict, preds

    def _get_train_measures(self, data, loss_dict, cache) -> dict:
        preds = cache
        _, targets = data
        measures = loss_dict
        for m in self.metrics:
            m_value = m(preds, targets)
            m_dict = get_dict(m_value, name=m._get_name())
            measures.update(m_dict)

        return measures

    def _get_eval_cache(self, data):
        inputs, targets = data

        if isinstance(inputs, dict):
            preds = self.models[0](**inputs)
        elif isinstance(inputs, (list, tuple)):
            preds = self.models[0](*inputs)
        else:
            preds = self.models[0](inputs)

        return preds, targets

    def _get_eval_logs(self, eval_caches):
        preds, trues = eval_caches

        loss = self.loss(preds, trues)
        loss_dict = get_dict(loss, prefix="Val_", name="Loss")

        measures = loss_dict
        for m in self.metrics:
            m_value = m(preds, trues)
            m_dict = get_dict(m_value, prefix="Val_", name=m._get_name())
            measures.update(m_dict)

        return measures
