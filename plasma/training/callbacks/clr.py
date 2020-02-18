import os

import numpy as np
import torch
import torch.optim as opts

from plasma.training.callbacks.base_class import Callback


class LrFinder(Callback):

    def __init__(self, min_lr, max_lr, epochs=3):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epochs = epochs

        self.scheduler = None
        self.gamma = 0
        self.history = {}

    def on_train_begin(self, **train_configs):
        epochs = self.epochs
        iterations = len(train_configs["train_loader"])

        for g in self.optimizer.param_groups:
            g["lr"] = self.min_lr

        self.gamma = (self.max_lr - self.min_lr) / (epochs * iterations)

    def on_training_batch_end(self, batch, x, y, pred, logs=None):
        for i, g in enumerate(self.optimizer.param_groups):
            if i in self.history:
                self.history[i].append((g["lr"], logs))
            else:
                self.history[i] = []

            g["lr"] += self.gamma

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.training = epoch + 1 < self.epochs

    def on_train_end(self):
        self.plot_data()

    def get_data(self, group=0, target="loss"):
        for lr, logs in self.history[group]:
            yield lr, logs[target]

    def plot_data(self, group=0, target="loss"):
        import matplotlib.pyplot as plt
        lrs, targets = zip(*self.get_data(group, target))

        plt.plot(lrs, targets)
        plt.xlabel("lr")
        plt.ylabel(target)
        plt.title(f"lr vs {target}")
        plt.show()


class CLR(Callback):

    def __init__(self, min_lr, max_lr, cycle_rate=2, reduce_lr_each_cycle=False):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr

        assert cycle_rate % 2 == 0, "cycle_rate must be divisible by 2"
        self.cycle_rate = cycle_rate
        self.reduce_lr_each_cycle = reduce_lr_each_cycle
        self.clr = None

    def on_train_begin(self, train_loader, **train_configs):
        iterations = len(train_loader)

        self.clr = opts.lr_scheduler.CyclicLR(self.optimizer, self.min_lr, self.max_lr,
                                              step_size_up=self.cycle_rate // 2 * iterations,
                                              mode="triangular2" if self.reduce_lr_each_cycle else "triangular")

    def on_training_batch_end(self, batch, x, y, pred, logs=None):
        self.clr.step()


class WarmRestart(Callback):

    def __init__(self, min_lr, t0=10, factor=2, cycles=3, lr_muls=None,
                 snapshot=True, directory="checkpoint", model_name=None):
        super().__init__()

        self.min_lr = min_lr
        self.t0 = t0
        self.factor = factor
        self.cycles = cycles
        self.lr_muls = [1] * (cycles - 1) if lr_muls is None else \
            lr_muls if type(lr_muls) in {list, tuple} else [lr_muls] * (cycles - 1)
        self.snapshot = snapshot
        self.dir = directory
        self.model_name = model_name or "model"

        self.base_lrs = None
        self.scheduler = None
        self.current_epoch = 0
        self.finished_cycles = 0
        self.max_epoch = t0

    def on_train_begin(self, **train_configs):
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]

        if not os.path.exists(self.dir) and self.snapshot:
            os.mkdir(self.dir)

    def on_epoch_begin(self, epoch):
        self.current_epoch += 1

        min_lr = self.min_lr
        for max_lr, g in zip(self.base_lrs, self.optimizer.param_groups):
            g["lr"] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(self.current_epoch / self.max_epoch * np.pi))

    def on_epoch_end(self, epoch, logs=None):
        for i, g in enumerate(self.optimizer.param_groups):
            logs[f"lr {i}"] = g["lr"]

        if self.current_epoch == self.max_epoch:
            self.current_epoch = 0
            self.max_epoch *= self.factor
            self.finished_cycles += 1

            print("starting cycle ", self.finished_cycles + 1) if self.finished_cycles != self.cycles else None
            if self.snapshot:
                model_state = self.model.state_dict()
                torch.save(model_state, f"{self.dir}/snapshot_{self.model_name}-{self.finished_cycles}.model")

                opt_state = self.optimizer.state_dict()
                torch.save(opt_state, f"{self.dir}/snapshot_{self.model_name}-{self.finished_cycles}.opt")

            lr_mul = self.lr_muls[self.finished_cycles - 1]
            self.base_lrs = [lr_mul * lr for lr in self.base_lrs]

        self.trainer.training = self.finished_cycles < self.cycles
