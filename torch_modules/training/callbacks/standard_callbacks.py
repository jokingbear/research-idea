import collections
import io
import os

import numpy as np
import torch
import torch.optim.lr_scheduler as schedulers
import csv

from torch_modules.training.callbacks.root_class import Callback
from torch.utils.tensorboard import SummaryWriter


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
                                                      patience=self.patience - 1, verbose=bool(self.verbose))

    def on_epoch_end(self, epoch, logs=None):
        for i, param_group in enumerate(self.optimizer.param_groups):
            logs[f"group {i} lr"] = param_group["lr"]

        self.scheduler.step(logs[self.monitor], epoch + 1)


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

    def __init__(self, file_path, monitor="val_loss", mode="min",
                 save_best_only=False, save_optimizer=False,
                 verbose=1):
        super().__init__()

        self.file_path = file_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_optimizer = save_optimizer
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
            print("saving model to ", self.file_path) if self.verbose else None
            torch.save(self.model.state_dict(), self.file_path + ".model")
            torch.save(self.optimizer.state_dict(), self.file_path + ".opt") if self.save_optimizer else None

            self.running_monitor_val = monitor_val


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


class Tensorboard(Callback):

    def __init__(self, log_dir, steps=50, flushes=60):
        super().__init__()

        self.log_dir = log_dir
        self.steps = steps
        self.current_step = 0
        self.flushes = flushes

        self.writer = None

    def on_train_begin(self):
        self.writer = SummaryWriter(self.log_dir, flush_secs=self.flushes)

    def on_batch_end(self, batch, logs=None):
        if self.current_step % self.steps == 0:
            for k in logs:
                print(k, "  ", logs[k])
                self.writer.add_scalar(k, logs[k], self.current_step)

        self.current_step += 1

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if "val" in k:
                self.writer.add_scalar(k, logs[k], epoch)

    def on_train_end(self):
        self.writer = None
