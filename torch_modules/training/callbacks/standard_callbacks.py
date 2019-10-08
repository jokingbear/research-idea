import collections
import io
import os

import numpy as np
import torch
import torch.optim.lr_scheduler as schedulers
import csv
import matplotlib.pyplot as plt
import imageio as iio

from torch_modules.training.callbacks.root_class import Callback
from collections import OrderedDict


class ReduceLROnPlateau(Callback):

    def __init__(self, monitor="val_loss", patience=5, mode="min", factor=0.1, verbose=True):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.factor = factor
        self.verbose = verbose
        self.scheduler = None

    def on_train_begin(self):
        self.scheduler = schedulers.ReduceLROnPlateau(self.optimizer, mode=self.mode, factor=self.factor,
                                                      patience=self.patience - 1, verbose=self.verbose)

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


class LRFinder(Callback):

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        """Calculate the learning rate."""
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        """Initialize the learning rate to the minimum value at the start of training."""
        self.optimizer.param_groups[0]["lr"] = self.min_lr

    def on_batch_end(self, epoch, logs=None):
        """Record previous batch statistics and update the learning rate."""
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(self.optimizer.param_groups[0]["lr"])
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.optimizer.param_groups[0]["lr"] = self.clr()

    def plot_lr(self):
        """Helper function to quickly inspect the learning rate schedule."""
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()

    def plot_loss(self):
        """Helper function to quickly observe the learning rate experiment results."""
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()


class GenImage(Callback):

    def __init__(self, path="gen_data", samples=1, rows=2, render_steps=50, render_size=(15, 15)):
        super().__init__()

        self.path = path
        self.samples = samples
        self.rows = rows
        self.render_steps = render_steps
        self.render_size = render_size

    def on_train_begin(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def on_batch_end(self, batch, logs=None):
        if batch % self.render_steps == 0:
            rows = self.rows
            g = self.model[1].eval()

            with torch.no_grad():
                imgs = g().cpu().numpy().transpose([0, 2, 3, 1])
                h, w, c = imgs.shape[1:]
                imgs = imgs.reshape([rows, -1, *imgs.shape[1:]]).transpose([0, 2, 1, 3, 4])
                imgs = imgs.reshape([rows * h, -1, c])
                imgs = imgs[..., 0] if imgs.shape[-1] == 1 else imgs

                _, ax = plt.subplots(figsize=self.render_size)
                ax.imshow(imgs)
                ax.axis("off")
                plt.show()

    def on_epoch_end(self, epoch, logs=None):
        g = self.model[1].eval()
        rows = self.rows

        path = f"{self.path}/epoch {epoch}"

        if not os.path.exists(path):
            os.mkdir(path)

        with torch.no_grad():
            for i in range(self.samples):
                imgs = g().cpu().numpy().transpose([0, 2, 3, 1])
                h, w, c = imgs.shape[1:]
                imgs = imgs.reshape([rows, -1, *imgs.shape[1:]]).transpose([0, 2, 1, 3, 4])
                imgs = imgs.reshape([rows * h, -1, c])
                imgs = imgs[..., 0] if imgs.shape[-1] == 1 else imgs

                iio.imsave(f"{path}/{i}.png", imgs)


class GanCheckpoint(Callback):

    def __init__(self, filename, overwrite=False):
        super().__init__()

        self.filename = filename
        self.overwrite = overwrite

    def on_epoch_end(self, epoch, logs=None):
        if self.overwrite:
            torch.save(self.model[-1].state_dict(), self.filename)
        else:
            torch.save(self.model[-1].state_dict(), f"{self.filename}-{epoch}")
