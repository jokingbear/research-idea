from abc import abstractmethod
from itertools import count
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..utils import eval_modules
from .utils import get_batch_tensors
from ..callbacks.standard_callbacks import ProgressBar


class BaseTrainer:

    def __init__(self, model: nn.Module, optimizer, loss: nn.Module, metrics=None, dtype=torch.float, rank=0):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.training = True
        self.dtype = dtype

        device = f'cuda:{rank}' if torch.cuda.device_count() > 0 else 'cpu'
        self.device = device
        self.rank = rank

    def fit(self, train_loader, valid_loader=None, callbacks=None, start_epoch=1):
        assert start_epoch > 0, "start epoch must be positive"
        callbacks = callbacks or []
        callbacks = [ProgressBar(), *callbacks]

        [c.set_trainer(self) for c in callbacks]
        
        train_configs = {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "start_epoch": start_epoch,
        }

        [c.on_train_begin(**train_configs) for c in callbacks]
        try:
            for e in count(start=start_epoch):
                [c.on_epoch_begin(e) for c in callbacks]

                train_logs = self._train_one_epoch(e, train_loader, callbacks)
                val_logs = {}

                if valid_loader is not None:
                    val_logs = self._evaluate_one_epoch(valid_loader, e, callbacks)

                logs = {**train_logs, **val_logs}

                [c.on_epoch_end(e, logs) for c in callbacks]

                if not self.training:
                    break

            [c.on_train_end(logs) for c in callbacks]
        except Exception as e:
            with open("trainer_error.txt", "w+") as handle:
                handle.write(str(e))
            raise

    def _train_one_epoch(self, epoch, train_loader, callbacks):
        logs = {}

        for i, data in enumerate(train_loader):
            data = self._extract_data(data)
            [c.on_training_batch_begin(epoch, i, data) for c in callbacks]

            self.model.train().zero_grad()
            loss_dict, caches = self._train_one_batch(data)

            with torch.no_grad():
                batch_logs = self._get_batch_logs(data, loss_dict, caches)
                logs['batch_logs'] = batch_logs

                [c.on_training_batch_end(epoch, i, data, caches, logs) for c in callbacks]

        del logs['batch_logs']
        return logs

    def _evaluate_one_epoch(self, test_loader, epoch=0, callbacks=()):
        eval_caches = []

        with eval_modules(*self.models):
            for i, data in enumerate(test_loader):
                data = self._extract_data(data)
                [c.on_validation_batch_begin(epoch, i, data) for c in callbacks]

                caches = self._get_eval_cache(data)
                eval_caches.append(caches)

                [c.on_validation_batch_end(epoch, i, data, caches) for c in callbacks]

            if torch.is_tensor(eval_caches[0]):
                eval_caches = torch.cat(eval_caches, dim=0)
            elif isinstance(eval_caches[0], (tuple, list)):
                n_pred = len(eval_caches[0])
                eval_caches = [torch.cat([c[i] for c in eval_caches], dim=0) for i in range(n_pred)]
            elif isinstance(eval_caches[0], dict):
                eval_caches = {k: torch.cat([c[k] for c in eval_caches], dim=0) for k in eval_caches[0]}
            else:
                raise 'only support tensor, tuple, list and dict cache'

            logs = self._get_eval_logs(eval_caches)
            return logs

    def _extract_data(self, batch_data):
        return get_batch_tensors(batch_data, self.dtype, self.device)

    @abstractmethod
    def _train_one_batch(self, data) -> Tuple[dict, object]:
        pass

    @abstractmethod
    def _get_batch_logs(self, data, loss_dict, cache) -> dict:
        pass

    @abstractmethod
    def _get_eval_cache(self, data):
        pass

    @abstractmethod
    def _get_eval_logs(self, eval_caches) -> dict:
        pass

    def extra_repr(self):
        return f"dtype={self.dtype}, device={self.device}"

    def __repr__(self):
        return f"{type(self).__name__}({self.extra_repr()})"

    def __str__(self):
        return repr(self)
