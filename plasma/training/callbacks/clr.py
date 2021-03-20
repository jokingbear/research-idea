import os
import numpy as np
import torch
import torch.optim as opts

from .base_class import Callback


class LrFinder(Callback):

    def __init__(self, min_lr=1e-5, max_lr=1, epochs=1, use_plotly=True):
        """
        Callback for finding task specific learning rate
        :param min_lr: learning rate lower bound
        :param max_lr: learning rate upper bound
        :param epochs: number of epoch to run
        :param use_plotly: use plotly in plot_data method
        """
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epochs = epochs
        self.use_plotly = use_plotly

        self.scheduler = None
        self.gamma = 0
        self.history = {}

    def on_train_begin(self, **train_configs):
        epochs = self.epochs
        iterations = len(train_configs["train_loader"])

        for g in self.optimizers[0].param_groups:
            g["lr"] = self.min_lr

        self.gamma = (self.max_lr - self.min_lr) / (epochs * iterations)

    def on_training_batch_end(self, epoch, step, inputs, targets, caches, logs=None):
        for i, g in enumerate(self.optimizers[0].param_groups):
            if i in self.history:
                self.history[i].append((g["lr"], logs))
            else:
                self.history[i] = []

            g["lr"] = g["lr"] + self.gamma

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.training = epoch + 1 < self.epochs

    def on_train_end(self):
        self.plot_data()

    def get_data(self, group=0, target="Loss"):
        for lr, logs in self.history[group]:
            yield lr, logs[target]

    def plot_data(self, group=0, target="Loss"):
        lrs, targets = zip(*self.get_data(group, target))

        if self.use_plotly:
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Scatter(x=lrs, y=targets))
            fig.update_layout(title=f"lr vs {target}", xaxis_title="lr", yaxis_title=target)
            fig.show("iframe")
        else:
            import matplotlib.pyplot as plt
            plt.plot(lrs, targets)
            plt.xlabel("lr")
            plt.ylabel(target)
            plt.title(f"lr vs {target}")
            plt.show()

    def extra_repr(self):
        return f"min_lr={self.min_lr}, max_lr={self.max_lr}, epochs={self.epochs}, use_plotly={self.use_plotly}"


class WarmRestart(Callback):

    def __init__(self, min_lr=0, t0=10, factor=2, cycles=3,
                 snapshot=True, directory="checkpoint", model_name=None):
        """
        :param min_lr: final learning
        :param t0: number of epoch in the 1st cycle
        :param factor: factor to change epoch each cycle
        :param cycles: number of cycle to run
        :param snapshot: whether to save snapshot each cycle
        :param directory: where to save the model
        :param model_name: model file name
        """
        super().__init__()

        self.min_lr = min_lr
        self.t0 = t0
        self.factor = factor
        self.cycles = cycles
        self.snapshot = snapshot
        self.dir = directory
        self.model_name = model_name or "warm"

        self.base_lrs = None
        self.scheduler = None
        self.run_epoch = 0
        self.finished_cycles = 0
        self.max_epoch = t0

    def on_train_begin(self, **train_configs):
        self.run_epoch = train_configs["start_epoch"] - 1
        train_loader = train_configs["train_loader"]
        self.scheduler = opts.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers[0],
                                                                       T_0=len(train_loader),
                                                                       T_mult=self.factor,
                                                                       eta_min=self.min_lr,
                                                                       last_epoch=self.run_epoch)

        if not os.path.exists(self.dir) and self.snapshot:
            os.mkdir(self.dir)

    def on_epoch_end(self, epoch, logs=None):
        self.run_epoch += 1

        for i, g in enumerate(self.optimizers[0].param_groups):
            logs[f"lr {i}"] = g["lr"]

        self.scheduler.step()

        if self.run_epoch == self.max_epoch:
            self.run_epoch = 0
            self.max_epoch *= self.factor
            self.finished_cycles += 1

            print("starting cycle ", self.finished_cycles + 1) if self.finished_cycles != self.cycles else None
            if self.snapshot:
                model_dict = self.models[0].state_dict()
                optim_dict = self.optimizers[0].state_dict()

                torch.save(model_dict, f"{self.dir}/snapshot_{self.model_name}_cycle_{self.finished_cycles}.model")
                torch.save(optim_dict, f"{self.dir}/snapshot_{self.model_name}_cycle_{self.finished_cycles}.optim")

        self.trainer.training = self.finished_cycles < self.cycles

    def extra_repr(self):
        return f"min_lr={self.min_lr}, t0={self.t0}, factor={self.factor}, cycles={self.cycles}, " \
               f"snapshot={self.snapshot}, directory={self.dir}, model_name={self.model_name}"


class SuperConvergence(Callback):

    def __init__(self, epochs, div_factor=25., final_div_factor=1e4,
                 snapshot=True, directory="checkpoint", model_name=None):
        """
        :param epochs: number of epoch to run
        :param snapshot: whether to take snapshot at the end of training
        :param directory: directory to put snapshot in
        :param model_name: name of the snapshot
        """
        super().__init__()

        self.epochs = epochs
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.snapshot = snapshot
        self.dir = directory
        self.name = model_name or "super_convergence"
        self.scheduler = None

    def on_train_begin(self, **train_configs):
        n = len(train_configs["train_loader"])
        max_lr = [g["lr"] for g in self.optimizers[0].param_groups]

        self.scheduler = opts.lr_scheduler.OneCycleLR(self.optimizers[0], max_lr,
                                                      epochs=self.epochs, steps_per_epoch=n,
                                                      div_factor=self.div_factor,
                                                      final_div_factor=self.final_div_factor)

        if not os.path.exists(self.dir) and self.snapshot:
            os.mkdir(self.dir)

    def on_training_batch_end(self, epoch, step, inputs, targets, caches, logs=None):
        self.scheduler.step()

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.training = epoch < self.epochs

    def on_train_end(self):
        model_dict = self.models[0].state_dict()
        torch.save(model_dict, f"{self.dir}/{self.name}.model")

    def extra_repr(self):
        return f"epochs={self.epochs}, div_factor={self.div_factor}, final_div_factor={self.final_div_factor}" \
               f"snapshot={self.snapshot}, directory={self.dir}, model_name={self.name}"


class Warmup(Callback):

    def __init__(self, init_lr, final_lr, n_epoch=1):
        super().__init__()

        assert init_lr > final_lr, "init_lr must be higher than final_lr"
        assert init_lr > 0, "init_lr must be bigger than 0"

        self.init_lr = init_lr
        self.final_lr = final_lr
        self.n_epoch = n_epoch

        self.lrs = []
        self.step = 0
        self.total_step = 0

    def on_train_begin(self, **train_configs):
        steps = len(train_configs["train_loader"])

        self.lrs = np.linspace(self.init_lr, self.final_lr, num=steps * self.n_epoch)
        self.total_step = steps * self.n_epoch

    def on_training_batch_begin(self, epoch, step, inputs, targets):
        if self.step < self.total_step:
            for g in self.optimizers[0].param_groups:
                g["lr"] = self.lrs[self.step]

    def on_training_batch_end(self, epoch, step, inputs, targets, caches, logs=None):
        self.step += 1

    def on_epoch_end(self, epoch, logs=None):
        self.trainer.training = self.step >= self.total_step
