from typing import Tuple

from .base_trainer import BaseTrainer
from .utils import get_dict


class StandardTrainer(BaseTrainer):
 
    def _train_one_batch(self, data) -> Tuple[dict, object]:
        inputs, targets = data
        preds = self.forward_func(inputs)
        loss = self.loss(preds, targets)

        if isinstance(loss, dict):
            loss_dict = {k: float(loss[k]) for k in loss}
            loss = loss["loss"]
        else:
            loss_dict = {"loss": float(loss)}

        loss.backward()
        self.optimizer.step()

        return loss_dict, preds

    def _get_batch_logs(self, data, loss_dict, pred_cache) -> dict:
        preds = pred_cache
        _, targets = data
        measures = loss_dict
        for m in self.metrics:
            m_value = m(preds, targets)
            m_dict = get_dict(m_value, name=m._get_name())
            measures.update(m_dict)

        return measures

    def _get_eval_cache(self, data):
        inputs, targets = data
        preds = self.forward_func(inputs)
        return preds, targets

    def _get_eval_logs(self, eval_caches):
        preds, trues = eval_caches

        loss = self.loss(preds, trues)
        loss_dict = get_dict(loss, prefix="val_", name="loss")

        measures = loss_dict
        for m in self.metrics:
            m_value = m(preds, trues)
            m_dict = get_dict(m_value, prefix="val_", name=m._get_name())
            measures.update(m_dict)

        return measures
