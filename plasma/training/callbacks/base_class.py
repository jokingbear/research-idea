from plasma.training.trainer import StandardTrainer as Trainer


class Callback:

    def __init__(self):
        self.trainer = None
        self.model = None
        self.optimizer = None
        self.training_config = None

    def on_train_begin(self, **train_configs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self):
        pass

    def set_trainer(self, trainer: Trainer):
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.trainer = trainer
