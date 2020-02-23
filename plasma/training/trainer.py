from itertools import count

import numpy as np
import torch
from torch.utils.data import DataLoader

import plasma.training.utils as utils


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
            workers=0, pin_memory=True, callbacks=None):
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

            val_logs = self.evaluate_one_epoch(test_loader, callbacks) if test is not None else {}

            logs = {**train_logs, **val_logs}

            [c.on_epoch_end(e, logs) for c in callbacks]

            if not self.training:
                break
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, callbacks):
        self.model.train()
        n = len(train)
        running_metrics = np.zeros(1 + len(self.metrics))
        metric_names = ["loss"] + [m.__name__ for m in self.metrics]

        with utils.get_tqdm()(total=n, desc="train") as pbar:
            for i, xy in enumerate(train):
                x, y = utils.get_inputs_labels(xy, self.x_type, self.x_device, self.y_type, self.y_device)
                [c.on_training_batch_begin(i, x, y) for c in callbacks]

                loss, pred = self.train_one_batch(x, y)

                with torch.no_grad():
                    running_metrics += [loss] + [m(pred, y) for m in self.metrics]
                    logs = dict(zip(metric_names, running_metrics / (i + 1)))

                    [c.on_training_batch_end(i, x, y, pred, logs) for c in callbacks]

                pbar.set_postfix(logs)
                pbar.update(1)

        return logs

    def train_one_batch(self, x, y):
        pred = self.model(x)
        loss = self.loss(pred, y)
        loss.backward()

        self.grad_step += 1

        if self.grad_step == self.grad_accumulation:
            self.grad_step = 0
            self.optimizer.step()
            self.model.zero_grad()

        return loss.detach(), pred.detach()

    def evaluate_one_epoch(self, test, callbacks):
        self.model.eval()
        n = len(test)

        preds = []
        trues = []
        with utils.get_tqdm()(total=n, desc="evaluate") as pbar, torch.no_grad():
            for i, xy in enumerate(test):
                x, y = utils.get_inputs_labels(xy, self.x_type, self.x_device, self.y_type, self.y_device)
                [c.on_validation_batch_begin(i, x, y) for c in callbacks]

                pred = self.model(x)
                preds.append(pred)
                trues.append(y)

                pbar.update(1)
                [c.on_validation_batch_end(i, x, y, pred) for c in callbacks]

            preds = torch.cat(preds, dim=0)
            trues = torch.cat(trues, dim=0)
            metric_logs = {m.__name__: float(m(preds, trues)) for m in self.metrics}
            val_logs = {"loss": float(self.loss(preds, trues)), **metric_logs}
            pbar.set_postfix(val_logs)

        return val_logs
