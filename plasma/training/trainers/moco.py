import torch
import torch.nn as nn
import torch.nn.functional as func

import numpy as np

from typing import Tuple
from .base_trainer import BaseTrainer
from ..losses import contrastive_loss_fn
from .utils import get_batch_tensors, get_dict


class MoCoTrainer(BaseTrainer):

    def __init__(self, query_encoder, key_encoder, optimizer, qsize=65536, t=0.2, momentum=0.999, normalize=True,
                 x_type=None, x_device=None):
        assert isinstance(query_encoder, nn.DataParallel), "query encoder needs to be data parallel"
        assert isinstance(key_encoder, nn.DataParallel), "key encoder needs to be data parallel"

        loss = nn.CrossEntropyLoss()
        metrics = [contrastive_loss_fn(t=t, normalize=normalize)]

        super().__init__([query_encoder, key_encoder], [optimizer], loss, metrics)

        self.qsize = qsize
        self.t = t
        self.momentum = momentum
        self.normalize = normalize
        self.devices = len(key_encoder.device_ids)

        self.x_type = x_type
        self.x_device = x_device
        self.queue = None
        self.queue_pointer = None

    def extract_data(self, batch_data):
        return get_batch_tensors(batch_data, self.x_type, self.x_device, self.x_type, self.x_device)

    def train_one_batch(self, inputs, targets) -> Tuple[dict, object]:
        queries = self.models[0](inputs)

        with torch.no_grad():
            shuffled_idc, inversed_idc = self._shuffle_indices(targets.shape[0])
            keys = self.models[1](targets[shuffled_idc])[inversed_idc]

        if self.normalize:
            queries = func.normalize(queries, p=2, dim=1)
            keys = func.normalize(keys, p=2, dim=1)

        # B x 1
        trues = (queries * keys).sum(dim=1, keepdim=True)

        if self.queue is None:
            queue = torch.rand(self.qsize, queries.shape[-1], device=queries.device)
            self.queue_pointer = 0
            self.queue = func.normalize(queue, dim=1)

        # B x qsize
        falses = func.linear(queries, self.queue)

        # B x (1 + qsize)
        logits = torch.cat([trues, falses], dim=1) / self.t
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = self.loss(logits, labels)
        loss.backward()
        self.optimizers[0].step()

        with torch.no_grad():
            self._update_key_encoder()
            self._update_queue(keys)

        return get_dict(loss), None

    def get_train_measures(self, inputs, targets, loss_dict, cache) -> dict:
        return loss_dict

    def get_eval_cache(self, inputs, targets):
        aug1 = self.models[0](inputs)
        aug2 = self.models[0](targets)

        return aug1, aug2

    def get_eval_logs(self, eval_caches) -> dict:
        pred1, pred2 = eval_caches
        logs = self.metrics[0](pred1, pred2)

        return get_dict(logs, prefix="val ")

    def _shuffle_indices(self, batch_size):
        shuffled_idc = np.random.choice(batch_size, size=batch_size, replace=False)
        mapping_idc = sorted([(old, new) for new, old in enumerate(shuffled_idc)], key=lambda kv: kv[0])
        inversed_idc = [new for _, new in mapping_idc]

        return shuffled_idc, inversed_idc

    def _update_queue(self, keys):
        batch_size = keys.shape[0]
        pointer = self.queue_pointer
        self.queue[pointer:pointer + batch_size] = keys
        self.queue_pointer = (pointer + batch_size) % self.qsize

    def _update_key_encoder(self):
        momentum = self.momentum
        for param_q, param_k in zip(self.models[0].parameters(), self.models[1].parameters()):
            param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data
