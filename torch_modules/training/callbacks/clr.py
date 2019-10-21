from torch_modules.training.callbacks import Callback


class LrFinder(Callback):

    def __init__(self, min_lr, max_lr, steps):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
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
        a = self.current_step / self.steps
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

    def __init__(self, min_lr, max_lr, epoch_steps, cycle_rate=2, reduce_lr_each_cycle=False):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.epoch_steps = epoch_steps
        self.current_step = 0

        assert cycle_rate % 2 == 0, "cycle_rate must be divisible by 2"
        self.cycle_rate = cycle_rate
        self.reduce_lr_each_cycle = reduce_lr_each_cycle

    def on_train_begin(self):
        for g in self.optimizer.param_groups:
            g["lr"] = self.min_lr

    def on_batch_end(self, batch, logs=None):
        self.current_step += 1

        lr = self.get_lr()

        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.cycle_rate == 0:
            self.current_step = 0

            if self.reduce_lr_each_cycle:
                self.max_lr = self.min_lr + (self.max_lr - self.min_lr) / 2
                print(f"reduced max lr to {self.max_lr}")

    def get_lr(self):
        a = (self.current_step % self.epoch_steps) / self.epoch_steps

        if self.current_step > self.epoch_steps * self.cycle_rate // 2:
            return self.max_lr - a * (self.max_lr - self.min_lr)
        else:
            return self.min_lr + a * (self.max_lr - self.min_lr)
