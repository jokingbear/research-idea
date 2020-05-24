import torch
import numpy as np
import pandas as pd

from itertools import count
from .utils import get_series, get_inputs_labels
from ..utils import get_tqdm


class ContrastiveTrainer:

    def __init__(self, model, optimizer, loss,
                 x_device=None, x_type=torch.float):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.x_device = x_device
        self.x_type = x_type

        self.training = True

    def fit(self, train_loader, test_loader=None, callbacks=None, start_epoch=1, evaluate_on_batch=False):
        assert start_epoch > 0, "start epoch must be positive"
        callbacks = callbacks or []

        [c.set_trainer(self) for c in callbacks]

        train_configs = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "start_epoch": start_epoch,
        }
        [c.on_train_begin(**train_configs) for c in callbacks]
        for e in count(start=start_epoch):
            print(f"epoch {e}")
            [c.on_epoch_begin(e) for c in callbacks]

            train_logs = self.train_one_epoch(train_loader, callbacks)

            if test_loader is not None:
                val_logs = self.evaluate_one_epoch(test_loader, callbacks, evaluate_on_batch)
            else:
                val_logs = {}

            logs = {**train_logs, **val_logs}

            [c.on_epoch_end(e, logs) for c in callbacks]

            if not self.training:
                break
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, callbacks):
        self.model.train()
        n = len(train)
        running_metrics = np.zeros([])

        with get_tqdm(total=n, desc="train") as pbar:
            for i, x in enumerate(train):
                x1, x2 = get_inputs_labels(x, self.x_type, self.x_device, self.x_type, self.x_device)
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
        with get_tqdm(total=len(test), desc="evaluate") as pbar, torch.no_grad():
            for i, x in enumerate(test):
                x1, x2 = get_inputs_labels(x, self.x_type, self.x_device, self.x_type, self.x_device)
                [c.on_validation_batch_begin(i, x1, x2) for c in callbacks]

                pred1 = self.model(x1)
                pred2 = self.model(x2)

                if on_batch:
                    losses = losses + get_series(self.loss(pred1, pred2), "val_")
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

                losses = get_series(self.loss(pred1s, pred2s), "val_")

            val_logs = losses
            pbar.set_postfix(val_logs)

        return val_logs
