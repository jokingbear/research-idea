class Callback:

    def __init__(self):
        self.trainer = None
        self.model = None
        self.optimizer = None

    def on_train_begin(self):
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

    def set_model_optimizer_trainer(self, model, optimizer, trainer):
        self.model = model
        self.optimizer = optimizer
        self.trainer = trainer
