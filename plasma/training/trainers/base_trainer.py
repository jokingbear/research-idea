from abc import abstractmethod
from itertools import count
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..utils import get_progress, eval_modules
from .utils import get_batch_tensors


class BaseTrainer:

    def __init__(self, models: List[nn.Module], optimizers, loss: nn.Module, metrics=None, types=(), devices=()):
        self.models = models
        self.optimizers = optimizers
        self.loss = loss
        self.metrics = metrics or []
        self.training = True
        self.types = types
        self.devices = devices

    def fit(self, train_loader, valid_loader=None, callbacks=None, start_epoch=1):
        assert start_epoch > 0, "start epoch must be positive"
        callbacks = callbacks or []

        [c.set_trainer(self) for c in callbacks]
        train_configs = {
            "train_loader": train_loader,
            "test_loader": valid_loader,
            "start_epoch": start_epoch,
        }
        [c.on_train_begin(**train_configs) for c in callbacks]
        try:
            for e in count(start=start_epoch):
                print(f"epoch {e}")
                [c.on_epoch_begin(e) for c in callbacks]

                train_logs = self._train_one_epoch(e, train_loader, callbacks)
                val_logs = {}

                if valid_loader is not None:
                    val_logs = self._evaluate_one_epoch(valid_loader, e, callbacks)

                logs = {**train_logs, **val_logs}

                [c.on_epoch_end(e, logs) for c in callbacks]

                if not self.training:
                    break

            [c.on_train_end() for c in callbacks]
        except Exception as e:
            with open("trainer_error.txt", "w+") as handle:
                handle.write(str(e))
            raise

    def _train_one_epoch(self, epoch, train_loader, callbacks):
        running_metrics = np.zeros([])

        with get_progress(total=len(train_loader), desc="train") as pbar:
            for i, data in enumerate(train_loader):
                data = self._extract_data(data)
                [c.on_training_batch_begin(epoch, i, data) for c in callbacks]

                [m.train().zero_grad() for m in self.models]
                loss_dict, caches = self._train_one_batch(data)

                with torch.no_grad():
                    measures = self._get_train_measures(data, loss_dict, caches)
                    measures = pd.Series(measures)

                    running_metrics = running_metrics + measures
                    logs = measures
                    [c.on_training_batch_end(epoch, i, data, caches, logs) for c in callbacks]

                    logs = logs.copy()
                    logs.update(running_metrics / (i + 1))

                pbar.set_postfix(logs)
                pbar.update()

        return logs

    def _evaluate_one_epoch(self, test_loader, epoch=0, callbacks=()):
        eval_caches = []

        with get_progress(total=len(test_loader), desc="eval") as pbar, eval_modules(*self.models):
            for i, data in enumerate(test_loader):
                data = self._extract_data(data)
                [c.on_validation_batch_begin(epoch, i, data) for c in callbacks]

                caches = self._get_eval_cache(data)
                eval_caches.append(caches)

                pbar.update()
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

            pbar.set_postfix(logs)
            return logs

    def _extract_data(self, batch_data):
        return get_batch_tensors(batch_data, self.types, self.devices)

    @abstractmethod
    def _train_one_batch(self, data) -> Tuple[dict, object]:
        pass

    @abstractmethod
    def _get_train_measures(self, data, loss_dict, cache) -> dict:
        pass

    @abstractmethod
    def _get_eval_cache(self, data):
        pass

    @abstractmethod
    def _get_eval_logs(self, eval_caches) -> dict:
        pass

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{type(self).__name__}({self.extra_repr()})"

    def __str__(self):
        return repr(self)
