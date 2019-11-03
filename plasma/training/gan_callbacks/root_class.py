from torch.utils.data import Dataset

from plasma.training.trainer.gan_trainer import GANTrainer


class Callback:

    def __init__(self):
        self.trainer = None
        self.discriminator = None
        self.generator = None
        self.dataset = None

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

    def set_trainer_dataset(self, trainer: GANTrainer, dataset: Dataset):
        self.trainer = trainer
        self.discriminator = trainer.discriminator
        self.generator = trainer.generator
        self.dataset = dataset
