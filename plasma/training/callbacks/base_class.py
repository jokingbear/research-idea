import torch.nn as nn


class Callback:

    def __init__(self):
        self.trainer = None
        self.model = None
        self.optimizer = None
        self.training_config = None

    def on_train_begin(self, **train_configs):
        pass

    def on_train_end(self, logs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_training_batch_begin(self, epoch, step, data):
        pass

    def on_training_batch_end(self, epoch, step, data, caches, logs):
        pass

    def on_validation_batch_begin(self, epoch, step, data):
        pass

    def on_validation_batch_end(self, epoch, step, data, caches):
        pass

    def set_trainer(self, trainer):
        if isinstance(trainer.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model = trainer.model.module
        else:
            self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.trainer = trainer

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{type(self).__name__}({self.extra_repr()})"

    def __str__(self):
        return repr(self)
