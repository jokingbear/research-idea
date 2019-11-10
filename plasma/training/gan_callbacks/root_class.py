from plasma.training.trainer.gan_trainer import GANTrainer


class Callback:

    def __init__(self):
        self.trainer = None
        self.discriminator = None
        self.generator = None
        self.d_optimizer = None
        self.g_optimizer = None
        self.dataset = None
        self.training_config = None

    def on_train_begin(self):
        pass

    def on_epoch_begin(self, e):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, e, logs=None):
        pass

    def on_train_end(self):
        pass

    def set_trainer(self, trainer: GANTrainer):
        self.trainer = trainer
        self.discriminator = trainer.discriminator
        self.generator = trainer.generator
        self.d_optimizer = trainer.d_optimizer
        self.g_optimizer = trainer.g_optimizer

    def set_training_config(self, **kwargs):
        self.training_config = kwargs
