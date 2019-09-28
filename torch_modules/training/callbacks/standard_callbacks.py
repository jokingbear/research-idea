import collections
import io
import os

import numpy as np
import torch
import torch.optim.lr_scheduler as schedulers
import csv

from torch_modules.training.callbacks.root_class import Callback
from collections import OrderedDict


class ReduceLROnPlateau(Callback):

    def __init__(self, monitor="val_loss", patience=5, mode="min", factor=0.1, verbose=1):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.factor = factor
        self.verbose = verbose > 0
        self.scheduler = None

    def on_train_begin(self):
        self.scheduler = schedulers.ReduceLROnPlateau(self.optimizer, mode=self.mode, factor=self.factor,
                                                      patience=self.patience - 1, verbose=False)

    def on_epoch_end(self, epoch, logs=None):
        lr0 = self.optimizer.param_groups[0]["lr"]

        self.scheduler.step(logs[self.monitor])
        lr1 = self.optimizer.param_groups[0]["lr"]

        if self.verbose and lr0 != lr1:
            print(f"reduce learning rate from {lr0} to {lr1}")

        logs["lr"] = lr1


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
        monitor_val = float(logs[self.monitor])

        is_max = self.monitor_val < monitor_val and self.mode == "max"
        is_min = self.monitor_val > monitor_val and self.mode == "min"

        reset_counter = is_max or is_min

        if reset_counter:
            print(f"{self.monitor} improved from {self.monitor_val} to {monitor_val}") if self.verbose else None
            self.monitor_val = monitor_val
            self.patience_count = 0
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

        is_save = not self.save_best_only or is_min or is_max

        if is_save:
            save_obj = self.model.state_dict() if self.save_weights_only else self.model

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

        self.parameters = {k: d[k].clone() for k in d}

    def on_batch_end(self, batch, logs=None):
        if batch % self.inner_step == 0 and batch != 0:
            kws0 = self.parameters
            kws1 = self.model.state_dict()
            alpha = self.alpha

            kws = [(k, kws0[k] + alpha * (kws1[k] - kws0[k])) for k in kws1]
            kws = OrderedDict(kws)

            self.parameters = kws
            self.model.load_state_dict(kws)

    def on_train_end(self):
        self.parameters = None


class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.

  Supports all values that can be represented as a string,
  including 1D iterables such as np.ndarray.

  Example:

  ```python
  csv_logger = CSVLogger('training.log')
  model.fit(X_train, Y_train, callbacks=[csv_logger])
  ```

  Arguments:
      filename: filename of the csv file, e.g. 'run/log.csv'.
      separator: string used to separate elements in the csv file.
      append: True: append if file exists (useful for continuing
          training). False: overwrite existing file,
  """

    def __init__(self, filename, separator=',', append=False):
        super().__init__()
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True

        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        self.csv_file = None

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, collections.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.trainer.train_mode:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
