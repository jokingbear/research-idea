from typing import Tuple

import numpy as np
import torch

from .base_trainer import BaseTrainer
from .utils import get_dict, get_batch_tensors


class ContrastiveTrainer(BaseTrainer):

    def __init__(self, model, optimizer, loss,
                 x_device=None, x_type=torch.float):
        super().__init__([model], [optimizer], loss)

        self.x_device = x_device
        self.x_type = x_type

    def extract_data(self, batch_data):
        return get_batch_tensors(batch_data, self.x_type, self.x_device, self.x_type, self.x_device)

    def train_one_batch(self, inputs, targets) -> Tuple[dict, object]:
        shuffle_idc = np.random.choice(targets.shape[0], size=targets.shape[0], replace=False)
        mapping_idc = [(old, new) for new, old in enumerate(shuffle_idc)]
        inverse_idc = [new for _, new in sorted(mapping_idc, key=lambda kv: kv[0])]

        aug1 = self.models[0](inputs)
        aug2 = self.models[0](targets[shuffle_idc])[inverse_idc]

        loss = self.loss(aug1, aug2)

        if isinstance(loss, dict):
            loss_dict = get_dict(loss, name="loss")
            loss = loss["loss"]
        else:
            loss_dict = {"loss": float(loss)}

        loss.backward()
        self.optimizers[0].step()

        return loss_dict, None

    def get_train_measures(self, inputs, targets, loss_dict, cache) -> dict:
        return loss_dict

    def get_eval_cache(self, inputs, targets):
        aug1 = self.models[0](inputs)
        aug2 = self.models[0](targets)

        return aug1, aug2

    def get_eval_logs(self, eval_caches) -> dict:
        aug1, aug2 = eval_caches

        loss = self.loss(aug1, aug2)
        return get_dict(loss, prefix="val_", name="loss")
