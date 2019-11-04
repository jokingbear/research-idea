import torch.optim as opts

from plasma.training.callbacks.root_class import Callback


class LrFinder(Callback):

    def __init__(self, min_lr, max_lr, epoch, iterations):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.epoch = epoch
        self.steps = iterations
        self.current_step = 0
        self.history = {}

    def on_train_begin(self):
        for g in self.optimizer.param_groups:
            g["lr"] = self.min_lr

    def on_batch_end(self, batch, logs=None):
        self.current_step += 1

        lr = self.get_lr()

        for i, g in enumerate(self.optimizer.param_groups):
            old_lr = g["lr"]
            g["lr"] = lr

            if i in self.history:
                self.history[i].append((old_lr, logs))
            else:
                self.history[i] = []

    def get_lr(self):
        a = self.current_step / self.steps / self.epoch
        lr = self.min_lr + a * (self.max_lr - self.min_lr)

        return lr

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
        plt.legend()
        plt.show()


class CLR(Callback):

    def __init__(self, min_lr, max_lr, iterations, cycle_rate=2, reduce_lr_each_cycle=False):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.iterations = iterations

        assert cycle_rate % 2 == 0, "cycle_rate must be divisible by 2"
        self.cycle_rate = cycle_rate
        self.reduce_lr_each_cycle = reduce_lr_each_cycle
        self.clr = None

    def on_train_begin(self):
        self.clr = opts.lr_scheduler.CyclicLR(self.optimizer, self.min_lr, self.max_lr,
                                              step_size_up=self.cycle_rate // 2 * self.iterations,
                                              mode="triangular2" if self.reduce_lr_each_cycle else "triangular")

    def on_batch_end(self, batch, logs=None):
        self.clr.step()


class WarmRestart(Callback):

    def __init__(self, min_lr, t0, factor=1):
        super().__init__()

        self.min_lr = min_lr
        self.t0 = t0
        self.factor = factor

        self.scheduler = None

    def on_train_begin(self):
        self.scheduler = opts.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.t0,
                                                                       T_mult=self.factor, eta_min=self.min_lr)

    def on_epoch_begin(self, epoch):
        self.scheduler.step(epoch)
