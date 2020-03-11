import torch
import numpy as np
import pandas as pd
import plasma.training.utils as utils

from torch.utils.data import DataLoader
from itertools import count


class StandardTrainer:

    def __init__(self, model, optimizer, loss, metrics=None,
                 x_device=None, x_type=torch.float, y_device=None, y_type=torch.long,
                 grad_accumulation=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []

        self.x_device = x_device
        self.x_type = x_type
        self.y_device = y_device
        self.y_type = y_type

        self.grad_accumulation = grad_accumulation or 1
        self.grad_step = 0

        self.training = True

    def fit(self, train, test=None, batch_size=32, val_batch_size=None,
            workers=0, pin_memory=True, callbacks=None, evaluate_on_batch=False):
        callbacks = callbacks or []
        val_batch_size = val_batch_size or batch_size

        train_sampler = train.get_sampler() if hasattr(train, "get_sampler") else None
        train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler, shuffle=train_sampler is None,
                                  drop_last=True, num_workers=workers, pin_memory=pin_memory)

        test_sampler = test.get_sampler() if hasattr(test, "get_sampler") else None
        test_loader = DataLoader(test, batch_size=val_batch_size, sampler=test_sampler, drop_last=False,
                                 num_workers=workers, pin_memory=pin_memory) if test is not None else None

        [c.set_trainer(self) for c in callbacks]
        [c.on_train_begin(train_loader=train_loader, test_loader=test_loader) for c in callbacks]
        for e in count(start=0):
            print(f"epoch {e + 1}")
            [c.on_epoch_begin(e) for c in callbacks]

            train_logs = self.train_one_epoch(train_loader, callbacks)

            val_logs = self.evaluate_one_epoch(test_loader, callbacks, evaluate_on_batch) if test is not None else {}

            logs = {**train_logs, **val_logs}

            [c.on_epoch_end(e, logs) for c in callbacks]

            if not self.training:
                break
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, callbacks):
        self.model.train()
        n = len(train)
        running_metrics = np.zeros([])

        with utils.get_tqdm()(total=n, desc="train") as pbar:
            for i, xy in enumerate(train):
                x, y = utils.get_inputs_labels(xy, self.x_type, self.x_device, self.y_type, self.y_device)
                [c.on_training_batch_begin(i, x, y) for c in callbacks]

                losses, pred = self.train_one_batch(x, y)

                with torch.no_grad():
                    metrics = pd.concat([losses, self.calculate_metrics(pred, y)])
                    logs = metrics

                    [c.on_training_batch_end(i, x, y, pred, logs) for c in callbacks]
                    running_metrics = running_metrics + metrics
                    logs = running_metrics / (i + 1)

                pbar.set_postfix(logs)
                pbar.update(1)

        return logs

    def train_one_batch(self, x, y):
        pred = self.model(x)
        loss = self.loss(pred, y)

        if isinstance(loss, dict):
            loss_series = pd.Series({k: float(loss[k]) for k in loss})
            loss = loss["loss"]
        else:
            loss_series = pd.Series({"loss": float(loss)})

        loss.backward()

        self.grad_step += 1

        if self.grad_step == self.grad_accumulation:
            self.grad_step = 0
            self.optimizer.step()
            self.model.zero_grad()

        return loss_series, pred.detach()

    def evaluate_one_epoch(self, test, callbacks, on_batch):
        self.model.eval()

        preds = []
        trues = []
        metrics = np.zeros([])
        losses = np.zeros([])
        with utils.get_tqdm()(total=len(test), desc="evaluate") as pbar, torch.no_grad():
            for i, xy in enumerate(test):
                x, y = utils.get_inputs_labels(xy, self.x_type, self.x_device, self.y_type, self.y_device)
                [c.on_validation_batch_begin(i, x, y) for c in callbacks]

                pred = self.model(x)

                if on_batch:
                    metrics = metrics + self.calculate_metrics(pred, y, "val_")
                    losses = losses + self.get_series(self.loss(pred, y), "val_")
                else:
                    preds.append(pred)
                    trues.append(y)

                pbar.update(1)
                [c.on_validation_batch_end(i, x, y, pred) for c in callbacks]

            if on_batch:
                metrics = metrics / len(test)
                losses = losses / len(test)
            else:
                if torch.is_tensor(preds):
                    preds = torch.cat(preds, dim=0)
                else:
                    col = len(preds[0])
                    preds = [torch.cat([p[c] for p in preds], dim=0) for c in range(col)]

                if torch.is_tensor(trues):
                    trues = torch.cat(trues, dim=0)
                else:
                    col = len(trues[0])
                    trues = [torch.cat([p[c] for p in preds], dim=0) for c in range(col)]

                metrics = self.calculate_metrics(preds, trues, "val_")
                losses = self.get_series(self.loss(preds, trues), "val_")

            val_logs = pd.concat([losses, metrics])
            pbar.set_postfix(val_logs)

        return val_logs

    def calculate_metrics(self, preds, trues, prefix=None):
        series = []
        for m in self.metrics:
            values = m(preds, trues)
            series.append(self.get_series(values, prefix, m.__name__))

        series = pd.concat(series, verify_integrity=True)

        return series

    def get_series(self, values, prefix=None, name=None):
        prefix = prefix or ""
        d = {prefix + k: float(values[k]) for k in values} if isinstance(values, dict) \
            else {prefix + (name or "loss"): float(values)}
        series = pd.Series(d)

        return series
