import numpy as np
import torch.optim.lr_scheduler as schedulers

from torch_modules.training.callbacks.root_class import Callback


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
                                                      patience=self.patience, verbose=self.verbose > 0)

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(logs[self.monitor])
        logs["lr"] = self.optimizer.param_groups[0]["lr"]
