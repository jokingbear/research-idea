import os

import torch
import torch.optim as opts

from plasma.training.callbacks.root_class import Callback


class LrFinder(Callback):

    def __init__(self, min_lr, max_lr):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.epochs = 1
        self.iterations = 1

        self.scheduler = None
        self.history = {}

    def set_train_config(self, epochs, iterations):
        self.epochs = epochs
        self.iterations = iterations

    def on_train_begin(self):
        self.scheduler = opts.lr_scheduler.CyclicLR(self.optimizer, self.min_lr, self.max_lr,
                                                    step_size_up=self.epochs * self.iterations)

    def on_batch_end(self, batch, logs=None):
        lr = self.scheduler.get_lr()

        self.scheduler.step()

        for i, g in enumerate(self.optimizer.param_groups):
            if i in self.history:
                self.history[i].append((lr, logs))
            else:
                self.history[i] = []

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.train_mode = epoch + 1 != self.epochs

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

        self.iterations = 1

        assert cycle_rate % 2 == 0, "cycle_rate must be divisible by 2"
        self.cycle_rate = cycle_rate
        self.reduce_lr_each_cycle = reduce_lr_each_cycle
        self.clr = None

    def set_train_config(self, epochs, iterations):
        self.iterations = iterations

    def on_train_begin(self):
        self.clr = opts.lr_scheduler.CyclicLR(self.optimizer, self.min_lr, self.max_lr,
                                              step_size_up=self.cycle_rate // 2 * self.iterations,
                                              mode="triangular2" if self.reduce_lr_each_cycle else "triangular")

    def on_batch_end(self, batch, logs=None):
        self.clr.step()


class WarmRestart(Callback):

    def __init__(self, min_lr, t0, factor=1, snapshot=False, directory="checkpoint", model_name=None):
        super().__init__()

        self.min_lr = min_lr
        self.t0 = t0
        self.factor = factor
        self.snapshot = snapshot
        self.dir = directory
        self.model_name = model_name or "model"

        self.scheduler = None
        self.current_epoch = 0
        self.max_epoch = t0

    def on_train_begin(self):
        self.scheduler = opts.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.t0,
                                                                       T_mult=self.factor, eta_min=self.min_lr)

        if not os.path.exists(self.dir) and self.snapshot:
            os.mkdir(self.dir)

    def on_epoch_begin(self, epoch):
        self.scheduler.step(epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1

        if self.current_epoch == self.max_epoch:
            self.current_epoch = 0
            self.max_epoch *= self.factor

            if self.snapshot:
                torch.save(self.model.state_dict(), f"{self.dir}/snapshot_{self.model_name}-{epoch + 1}")
