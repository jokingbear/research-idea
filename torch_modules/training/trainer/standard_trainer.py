import numpy as np
import torch
import torch_modules.training.trainer.utils as utils

from torch.utils.data import DataLoader, RandomSampler as Sampler


class StandardTrainer:

    def __init__(self, model, optimizer, loss, metrics=None,
                 x_device=None, x_type=torch.float, y_device=None, y_type=torch.long):
        self.model = model.train()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []

        self.x_device = x_device
        self.x_type = x_type
        self.y_device = y_device
        self.y_type = y_type

        self.train_mode = True

    def fit(self, train, test=None, epochs=1, batch_size=32, val_batch_size=1,
            workers=0, pin_memory=True, callbacks=None):
        callbacks = callbacks or []

        [c.set_model_optimizer_trainer(self.model, self.optimizer, self) for c in callbacks]
        [c.on_train_begin() for c in callbacks]

        train_loader = DataLoader(train, batch_size=batch_size, sampler=train if isinstance(train, Sampler) else None,
                                  shuffle=not isinstance(train, Sampler), drop_last=True,
                                  num_workers=workers, pin_memory=pin_memory)

        test_loader = DataLoader(test, batch_size=val_batch_size, sampler=test if isinstance(test, Sampler) else None,
                                 drop_last=True, num_workers=workers, pin_memory=pin_memory) if test else None

        for e in range(epochs):
            print(f"epochs: {e + 1}/{epochs}")

            [c.on_epoch_begin(e) for c in callbacks]

            train_logs = self.train_one_epoch(train_loader, callbacks)

            val_logs = self.evaluate_one_epoch(test_loader) if test else {}

            logs = {**train_logs, **val_logs}

            [c.on_epoch_end(e, logs) for c in callbacks]

            if not self.train_mode:
                break

        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, callbacks=None):
        self.model.train()
        n = len(train)
        callbacks = callbacks or []
        metrics_names = ["loss"] + [m.__name__ for m in self.metrics]
        running_metrics = np.zeros(shape=len(self.metrics) + 1)

        with utils.get_pbar()(total=n, desc="train") as pbar:
            for i, (x, y) in enumerate(train):
                [c.on_batch_begin(i) for c in callbacks]

                x = utils.to_device(x, self.x_type, self.x_device)
                y = utils.to_device(y, self.y_type, self.y_device, return_array=False)

                loss, y_pred = self.train_one_batch(*x, y)

                with torch.no_grad():
                    current_metrics = self.get_metrics(loss, y_pred, y)
                    running_metrics += current_metrics

                    logs = dict(zip(metrics_names, running_metrics / (i + 1)))

                    [c.on_batch_end(i, logs) for c in callbacks]

                pbar.set_postfix(logs, refresh=False)
                pbar.update(1)

        return logs

    def train_one_batch(self, *x, y):
        self.model.zero_grad()
        y_pred = self.model(*x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.detach(), y_pred.detach()

    def evaluate_one_epoch(self, test):
        self.model.eval()
        n = len(test)
        metrics_names = ["val_loss"] + ["val_" + m.__name__ for m in self.metrics]
        running_metrics = np.zeros(len(self.metrics) + 1)

        with utils.get_pbar()(total=n, desc="evaluate") as pbar, torch.no_grad():
            for i, (x, y) in enumerate(test):
                x = utils.to_device(x, self.x_type, self.x_device)
                y = utils.to_device(y, self.y_type, self.y_device, return_array=False)

                y_pred = self.model(*x)
                loss = self.loss(y_pred, y)

                metrics = self.get_metrics(loss, y_pred, y)
                running_metrics += metrics
                pbar.update(1)

            val_logs = dict(zip(metrics_names, running_metrics / n))
            pbar.set_postfix(val_logs)

        return val_logs

    def get_metrics(self, loss, y_pred, y):
        metrics = [float(m(y_pred, y)) for m in self.metrics]
        metrics = [float(loss)] + metrics

        return np.array(metrics)
