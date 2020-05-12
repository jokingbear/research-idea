import torch
import numpy as np
import pandas as pd
import plasma.training.utils as utils

from torch.utils.data import DataLoader
from itertools import count


class ContrastiveTrainer:

    def __init__(self, model, optimizer, loss,
                 x_device=None, x_type=torch.float):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.x_device = x_device
        self.x_type = x_type

        self.training = True

    def fit(self, train, test=None, batch_size=32, val_batch_size=None,
            workers=0, pin_memory=True, callbacks=None, start_epoch=1, evaluate_on_batch=False):
        callbacks = callbacks or []
        val_batch_size = val_batch_size or batch_size

        train_sampler = train.get_sampler() if hasattr(train, "get_sampler") else None
        train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler, shuffle=train_sampler is None,
                                  drop_last=True, num_workers=workers, pin_memory=pin_memory)

        test_loader = DataLoader(test, batch_size=val_batch_size, shuffle=False, drop_last=False,
                                 num_workers=workers, pin_memory=pin_memory) if test is not None else None

        [c.set_trainer(self) for c in callbacks]
        [c.on_train_begin(train_loader=train_loader, test_loader=test_loader) for c in callbacks]
        for e in count(start=start_epoch):
            print(f"epoch {e}")
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
            for i, x in enumerate(train):
                x1, x2 = utils.get_inputs_labels(x, self.x_type, self.x_device, self.x_type, self.x_device)
                [c.on_training_batch_begin(i, x1, x2) for c in callbacks]

                loss = self.train_one_batch(x1, x2)

                with torch.no_grad():
                    logs = loss.copy()

                    [c.on_training_batch_end(i, x1, x2, None, logs) for c in callbacks]
                    running_metrics = running_metrics + loss
                    logs = logs.copy()
                    logs.update(running_metrics / (i + 1))

                pbar.set_postfix(logs)
                pbar.update(1)

        return logs

    def train_one_batch(self, x1, x2):
        self.model.zero_grad()

        shuffle_idc = np.random.choice(x1.shape[0], size=x1.shape[0], replace=False)
        mapping_idc = [(old, new) for new, old in enumerate(shuffle_idc)]
        inverse_idc = [new for _, new in sorted(mapping_idc, key=lambda kv: kv[0])]

        pred1 = self.model(x1)
        pred2 = self.model(x2[shuffle_idc])[inverse_idc]
        loss = self.loss(pred1, pred2)

        if isinstance(loss, dict):
            loss_series = pd.Series({k: float(loss[k]) for k in loss})
            loss = loss["loss"]
        else:
            loss_series = pd.Series({"loss": float(loss)})

        loss.backward()

        self.optimizer.step()

        return loss_series

    def evaluate_one_epoch(self, test, callbacks, on_batch):
        self.model.eval()

        pred1s = []
        pred2s = []
        losses = np.zeros([])
        with utils.get_tqdm()(total=len(test), desc="evaluate") as pbar, torch.no_grad():
            for i, x in enumerate(test):
                x1, x2 = utils.get_inputs_labels(x, self.x_type, self.x_device, self.x_type, self.x_device)
                [c.on_validation_batch_begin(i, x1, x2) for c in callbacks]

                pred1 = self.model(x1)
                pred2 = self.model(x2)

                if on_batch:
                    losses = losses + self.get_series(self.loss(pred1, pred2), "val_")
                else:
                    pred1s.append(pred1)
                    pred2s.append(pred2)

                pbar.update(1)
                [c.on_validation_batch_end(i, x1, x2, None) for c in callbacks]

            if on_batch:
                losses = losses / len(test)
            else:
                if torch.is_tensor(pred1s[0]):
                    pred1s = torch.cat(pred1s, dim=0)
                else:
                    col = len(pred1s[0])
                    pred1s = [torch.cat([p[c] for p in pred1s], dim=0) for c in range(col)]

                if torch.is_tensor(pred2s[0]):
                    pred2s = torch.cat(pred2s, dim=0)
                else:
                    col = len(pred2s[0])
                    pred2s = [torch.cat([p[c] for p in pred2s], dim=0) for c in range(col)]

                losses = self.get_series(self.loss(pred1s, pred2s), "val_")

            val_logs = losses
            pbar.set_postfix(val_logs)

        return val_logs

    def get_series(self, values, prefix=None, name=None):
        prefix = prefix or ""
        d = {prefix + k: float(values[k]) for k in values} if isinstance(values, dict) \
            else {prefix + (name or "loss"): float(values)}
        series = pd.Series(d)

        return series
