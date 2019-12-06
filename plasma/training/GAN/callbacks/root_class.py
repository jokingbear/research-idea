from plasma.training.GAN.trainer import GANTrainer


class Callback:

    def __init__(self):
        self.trainer = None
        self.discriminator = None
        self.generator = None
        self.discriminator_optimizer = None
        self.generator_optimizer = None

    def on_train_begin(self, **train_configs):
        pass

    def on_epoch_begin(self, e):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_epoch_end(self, e, logs):
        pass

    def on_train_end(self):
        pass

    def set_trainer(self, trainer: GANTrainer):
        self.trainer = trainer
        self.discriminator = trainer.discriminator
        self.generator = trainer.generator
        self.discriminator_optimizer = trainer.discriminator_optimizer
        self.generator_optimizer = trainer.generator_optimizer
