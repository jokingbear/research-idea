import torch
import torch.nn as nn
import torch.nn.functional as func

import numpy as np

from typing import Tuple
from .base_trainer import BaseTrainer
from ..losses import contrastive_loss_fn
from .utils import get_inputs_labels


class MocoTrainer(BaseTrainer):

    def __init__(self, encoder_base, model_kwargs, optimizer, qsize=65536, t=0.1, momentum=0.999,
                 normalize=True, device_ids=(0, 1), x_type=None, x_device=None):
        encoder_q = encoder_base(**model_kwargs)
        encoder_k = encoder_base(**model_kwargs)

        encoder_q = nn.DataParallel(encoder_q, device_ids=device_ids)
        encoder_k = nn.DataParallel(encoder_k, device_ids=device_ids)

        loss = nn.CrossEntropyLoss()
        metrics = [contrastive_loss_fn(t=t)]

        for p_q, p_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            p_q.data.copy_(p_k.data)

        super(MocoTrainer, self).__init__([encoder_q, encoder_k], [optimizer], loss, metrics)

        self.qsize = qsize
        self.t = t
        self.momentum = momentum
        self.normalize = normalize
        self.devices = len(device_ids)

        self.x_type = x_type
        self.x_device = x_device
        self.queue = [for _ in range()]

    def extract_data(self, batch_data):
        return get_inputs_labels(batch_data, self.x_type, self.x_device, self.x_type, self.x_device)

    def train_one_batch(self, inputs, targets) -> Tuple[dict, object]:
        q = self.models[0](inputs)

        with torch.no_grad():
            shuffled_idc, inversed_idc = self.shuffle_indices(targets.shape[0])
            k = self.models[1](targets[shuffled_idc])[inversed_idc]

        if self.normalize:
            q = func.normalize(q, dim=1)
            k = func.normalize(k, dim=1)

        true = (q * k).sum(dim=1, keepdim=True)

        if self.queue is None:
            false =
        pass

    def get_train_measures(self, inputs, targets, loss_dict, cache) -> dict:
        pass

    def get_eval_cache(self, inputs, targets):
        pass

    def get_eval_logs(self, eval_caches) -> dict:
        pass

    def shuffle_indices(self, batch_size):
        idc = np.arange(0, batch_size)
        divide_device = idc.reshape([self.devices, -1])

        shuffle_device_idc = np.random.choice(self.devices, size=self.devices, replace=False)
        mapping_device_idc = sorted([(old, new) for new, old in enumerate(shuffle_device_idc)], key=lambda kv: kv[0])
        inverse_device_idc = [new for _, new in mapping_device_idc]

        shuffled_idc = divide_device[shuffle_device_idc].reshape([-1])
        inversed_idc = divide_device[inverse_device_idc].reshape([-1])

        return shuffled_idc, inversed_idc
