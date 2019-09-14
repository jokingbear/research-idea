import numpy as np
import torch
import torch.optim.lr_scheduler as schedulers

from torch_modules.training.callbacks.root_class import Callback
from collections import OrderedDict


class ReduceLROnPlateau(Callback):

    def __init__(self, monitor="val_loss", patience=5, mode="min", factor=0.1, verbose=1):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.factor = factor
        self.verbose = verbose
        self.scheduler = None

    def on_train_begin(self):
        self.scheduler = schedulers.ReduceLROnPlateau(self.optimizer, mode=self.mode, factor=self.factor,
                                                      patience=self.patience, verbose=self.verbose > 0)

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(logs[self.monitor])
        lr = self.optimizer.param_groups[0]["lr"]

        logs["lr"] = lr


class EarlyStopping(Callback):

    def __init__(self, monitor="val_loss", patience=10, mode="min", verbose=1):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.monitor_val = None
        self.patience_count = 0

    def on_train_begin(self):
        self.monitor_val = np.inf if self.mode == "min" else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        monitor_val = float(logs[self.monitor_val])

        if self.monitor_val < monitor_val and self.mode == "max":
            self.monitor_val = monitor_val
        elif self.monitor_val > monitor_val and self.mode == "min":
            self.monitor_val = monitor_val
        else:
            self.patience_count += 1
            print(f"model didn't improve from {self.monitor_val:.04f}") if self.verbose else None

        if self.patience_count == self.patience:
            self.trainer.train_mode = False
            print("Early stopping") if self.verbose else None


class ModelCheckpoint(Callback):

    def __init__(self, filepath, monitor="val_loss", mode="min", save_best_only=False, save_weights_only=False,
                 verbose=1):
        super().__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.running_monitor_val = None

    def on_train_begin(self):
        self.running_monitor_val = np.inf if self.mode == "min" else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        monitor_val = logs[self.monitor]

        is_max = self.running_monitor_val < monitor_val and self.mode == "max"
        is_min = self.running_monitor_val > monitor_val and self.mode == "min"

        inform = self.verbose and (is_min or is_max)

        if inform:
            print(f"{self.monitor} improved from {self.running_monitor_val} to {monitor_val}")

        is_save = not self.save_best_only or is_min or is_max

        if is_save:
            save_obj = self.model.state_dict if self.save_weights_only else self.model

            print("saving model to ", self.filepath) if self.verbose else None
            torch.save(save_obj, self.filepath)

            self.running_monitor_val = monitor_val


class Lookahead(Callback):

    def __init__(self, alpha=0.5, inner_step=5):
        super().__init__()

        self.alpha = alpha
        self.inner_step = inner_step
        self.parameters = None

    def on_train_begin(self):
        d = self.model.state_dict()

        self.parameters = [d[k].numpy() for k in d]

    def on_batch_end(self, batch, logs=None):
        if batch % self.inner_step == 0 and batch != 0:
            w0s = self.parameters
            p1 = self.model.state_dict()
            w1s = [p1[k].numpy() for k in p1]
            alpha = self.alpha

            ws = [w0 + alpha * (w1 - w0) for w0, w1 in zip(w0s, w1s)]
            keys = p1.keys()

            self.parameters = ws
            ws = [torch.tensor(w) for w in ws]
            parameters = OrderedDict(zip(keys, ws))
            self.model.load_state_dict(parameters)
