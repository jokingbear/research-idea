import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import *


class LRFinder(Callback):
    """
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    """

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
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        """Record previous batch statistics and update the learning rate."""
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

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


class SGDRScheduler(Callback):
    """Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    """
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2,
                 save_on_reset=False,
                 prefix="Model"):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.save_on_reset = save_on_reset
        self.prefix = prefix

        self.history = {}

        super().__init__()

    def clr(self):
        """Calculate the learning rate."""
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs=None):
        """Initialize the learning rate to the minimum value at the start of training."""
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs=None):
        """Record previous batch statistics and update the learning rate."""
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs=None):
        """Check for end of current cycle, apply restarts when necessary."""
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

            if self.save_on_reset:
                self.model.save_weights(f"{self.prefix}-{epoch}")

    def on_train_end(self, logs=None):
        """Set weights to the values from the end of the most recent cycle for best performance."""
        self.model.set_weights(self.best_weights)


class FunctionalLRScheduler(Callback):

    def __init__(self, schedule_fn, verbose=1, monitor="val_loss"):
        self.schedule_fn = schedule_fn
        self.verbose = verbose
        self.monitor = monitor

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        lr = K.get_value(self.model.optimizer.lr)
        monitor_val = logs[self.monitor]

        new_lr = self.schedule_fn(epoch, lr, monitor_val)

        if self.verbose:
            print("Update learning rate to ", new_lr)

        K.set_value(self.model.optimizer.lr, new_lr)

        logs["lr"] = new_lr


class Lookahead(Callback):

    def __init__(self, alpha=0.5, inner_step=5):
        super().__init__()

        self.alpha = alpha
        self.inner_step = inner_step
        self.weights = None

    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()

    def on_batch_end(self, batch, logs=None):
        self._look_ahead() if batch % self.inner_step == 0 and batch != 0 else None

    def _look_ahead(self):
        w0s = self.weights
        w1s = self.model.get_weights()
        alpha = self.alpha

        self.weights = [w0 + alpha * (w1 - w0) for w0, w1 in zip(w0s, w1s)]
        self.model.set_weights(self.weights)
