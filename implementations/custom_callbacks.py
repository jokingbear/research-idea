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


class RandomSearchLR(Callback):

    def __init__(self, min_lr=1E-5, max_lr=1E-2, n_epoch=3):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_epoch = n_epoch
        self.init_weights = None

    def on_train_begin(self, logs=None):
        self.init_weights = self.model.get_weights()

    def on_epoch_begin(self, epoch, logs=None):
        lr = np.random.uniform(self.min_lr, self.max_lr)
        K.set_value(self.model.optimizer.lr, lr)
        print("set lr to ", lr)
        print("resetting model's weight")
        self.model.set_weights(self.init_weights)
