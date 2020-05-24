import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from abc import abstractmethod
from itertools import count
from ..utils import get_tqdm, eval_modules
from typing import List, Tuple


class BaseTrainer:

    def __init__(self, models: List[nn.Module], optimizers, loss, metrics=None):
        self.models = models
        self.optimizers = optimizers
        self.loss = loss
        self.metrics = metrics or []
        self.training = True

    def fit(self, train_loader, valid_loader=None, callbacks=None, start_epoch=1, evaluate_on_batch=False):
        assert start_epoch > 0, "start epoch must be positive"
        callbacks = callbacks or []

        [c.set_trainer(self) for c in callbacks]

        train_configs = {
            "train_loader": train_loader,
            "test_loader": valid_loader,
            "start_epoch": start_epoch,
        }
        [c.on_train_begin(**train_configs) for c in callbacks]
        for e in count(start=start_epoch):
            print(f"epoch {e}")
            [c.on_epoch_begin(e) for c in callbacks]

            train_logs = self.train_one_epoch(e, train_loader, callbacks)
            val_logs = pd.Series({})

            if valid_loader is not None:
                val_logs = self.evaluate_one_epoch(valid_loader, callbacks, evaluate_on_batch)

            logs = val_logs.combine_first(train_logs)

            [c.on_epoch_end(e, logs) for c in callbacks]

            if not self.training:
                break
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, epoch, train_loader, callbacks) -> pd.Series:
        running_metrics = np.zeros([])

        with get_tqdm(len(train_loader), "train") as pbar:
            for i, data in enumerate(train_loader):
                [m.train() for m in self.models]

                inputs, targets = self.extract_data(data)
                [c.on_training_batch_begin(epoch, i, inputs, targets) for c in callbacks]

                loss_dict, caches = self.train_one_batch(inputs, targets)

                with torch.no_grad():
                    measures = self.get_train_measures(inputs, targets, loss_dict, caches)
                    measures = pd.Series(measures)

                    running_metrics = running_metrics + measures
                    logs = measures
                    [c.on_training_batch_end(epoch, i, inputs, targets, caches, logs) for c in callbacks]

                    logs = logs.copy()
                    logs.update(running_metrics / (i + 1))

                pbar.set_postfix(logs)
                pbar.update()

        return logs

    def evaluate_one_epoch(self, test_loader, epoch=0, callbacks=()) -> pd.Series:
        eval_caches = []

        with get_tqdm(len(test_loader), "eval") as pbar, eval_modules(self.models):
            for i, data in enumerate(test_loader):
                inputs, targets = self.extract_data(data)
                [c.on_validation_batch_begin(epoch, i, inputs, inputs) for c in callbacks]

                caches = self.get_eval_cache(inputs, targets)
                eval_caches.append(caches)

                pbar.update(1)
                [c.on_validation_batch_end(epoch, i, inputs, targets, caches) for c in callbacks]

        if torch.is_tensor(eval_caches[0]):
            eval_caches = torch.cat(eval_caches, dim=0)
        else:
            n_pred = len(eval_caches[0])
            eval_caches = [torch.cat([c[i] for c in eval_caches], dim=0) for i in range(n_pred)]

        logs = self.get_eval_logs(eval_caches)

        pbar.set_postfix(logs)
        return logs

    @abstractmethod
    def extract_data(self, batch_data):
        pass

    @abstractmethod
    def train_one_batch(self, inputs, targets) -> Tuple[dict, object]:
        pass

    @abstractmethod
    def get_preds_trues(self, inputs, targets):
        pass

    @abstractmethod
    def get_train_measures(self, inputs, targets, loss_dict, cache) -> dict:
        pass

    @abstractmethod
    def get_eval_cache(self, inputs, targets):
        pass

    @abstractmethod
    def get_eval_logs(self, eval_caches) -> pd.Series:
        pass
