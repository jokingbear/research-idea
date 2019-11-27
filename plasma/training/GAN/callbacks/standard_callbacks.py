import collections
import os

import torch.utils.data as data
import torchvision as tv

from plasma.training.GAN.callbacks.root_class import Callback


class GenImage(Callback):

    def __init__(self, save_dir="gen_image", steps=50, rows=8, normalize=True):
        super().__init__()

        self.save_dir = save_dir
        self.steps = steps
        self.rows = rows
        self.normalize = normalize

    def on_train_begin(self, **train_configs):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if not os.path.exists(self.save_dir + "/iterations"):
            os.mkdir(self.save_dir + "/iterations")

        if not os.path.exists(self.save_dir + "/epochs"):
            os.mkdir(self.save_dir + "/epochs")

    def on_batch_end(self, batch, logs):
        if batch % self.steps == 0:
            self.save_image(f"{self.save_dir}/iterations/{batch}.png")

    def on_epoch_begin(self, e):
        self.save_image(f"{self.save_dir}/epochs/{e}.png")

    def save_image(self, filename):
        imgs = self.generator(**self.trainer.g_kwargs)
        tv.utils.save_image(imgs, filename, nrow=self.rows, normalize=self.normalize)


class Progressive(Callback):

    def __init__(self, min_resolution, max_resolution, default_epochs, epoch_dict=None,
                 batch_dict=None, discriminator_lr_dict=None, generator_lr_dict=None):
        super().__init__()

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

        self.default_epochs = default_epochs
        self.epoch_dict = epoch_dict or {}
        self.batch_dict = batch_dict or {}
        self.discriminator_lr_dict = discriminator_lr_dict or {}
        self.generator_lr_dict = generator_lr_dict or {}

        self.iterations = 0
        self.epoch_counter = 0
        self.max_epoch = 0
        self.loader = None
        self.fade = False

    def on_train_begin(self, **train_configs):
        loader = train_configs["loader"]
        self.loader = loader
        self.iterations = len(loader)

        self.set_config()

    def on_batch_begin(self, batch):
        if self.fade:
            alpha = (batch + 1) * (self.epoch_counter + 1) / self.iterations / self.max_epoch
        else:
            alpha = 1

        self.trainer.g_kwargs["alpha"] = alpha

    def on_epoch_end(self, e, logs):
        self.epoch_counter += 1

        if self.epoch_counter == self.max_epoch:
            if not self.fade:
                self.min_resolution *= 2
                print("increase resolution to ", self.min_resolution)

            self.fade = not self.fade
            self.set_config()

    def set_config(self):
        resolution = self.min_resolution
        self.loader.dataset.resolution = resolution

        if resolution in self.batch_dict:
            batch_size = self.batch_dict[resolution]
            self.loader.batch_sampler = data.sampler.BatchSampler(self.loader.sampler, batch_size, drop_last=True)
            self.trainer.g_kwargs["batch_size"] = batch_size

        if self.fade:
            self.discriminator_optimizer.state = collections.defaultdict(dict, {})
            self.generator_optimizer.state = collections.defaultdict(dict, {})

        if resolution in self.discriminator_lr_dict:
            for lr, g in zip(self.discriminator_lr_dict[resolution], self.discriminator_optimizer.param_groups):
                g["lr"] = lr

        if resolution in self.generator_lr_dict:
            for lr, g in zip(self.generator_lr_dict[resolution], self.generator_optimizer.param_groups):
                g["lr"] = lr

        self.epoch_counter = 0
        if resolution in self.epoch_dict:
            self.max_epoch = self.epoch_dict[resolution]
