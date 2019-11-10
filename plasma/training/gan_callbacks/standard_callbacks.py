import os

import torch
import torchvision as tv

from plasma.training.gan_callbacks.root_class import Callback


class GenImage(Callback):

    def __init__(self, path="gen_data", rows=4, steps=50, normalize=True):
        super().__init__()

        self.path = path
        self.rows = rows
        self.steps = steps
        self.normalize = normalize

    def on_train_begin(self):
        if not os.path.exists(f"{self.path}"):
            os.mkdir(self.path)

        if not os.path.exists(f"{self.path}/iterations"):
            os.mkdir(self.path + "/iterations")

        if not os.path.exists(f"{self.path}/epochs"):
            os.mkdir(self.path + "/epochs")

    def on_batch_end(self, batch, logs=None):
        if batch % self.steps == 0:
            self.render_image(f"{self.path}/iterations/{batch}.png")

    def on_epoch_end(self, e, logs=None):
        self.render_image(f"{self.path}/epochs/{e}.png")

    def render_image(self, file_name):
        g = self.generator.eval()
        imgs = g(**self.trainer.g_kwargs)
        tv.utils.save_image(imgs, file_name, nrow=self.rows, normalize=self.normalize)


# TODO: change this flow
class Progressive(Callback):

    def __init__(self, iterations, start_epochs=30, increase_rate=1.5, verbose=1):
        super().__init__()

        assert increase_rate >= 1, "increase rate must be bigger than or equal to 1"

        self.iterations = iterations
        self.epochs = start_epochs
        self.increase_rate = increase_rate
        self.verbose = verbose

        self.epoch_counter = 0
        self.fade_in = False
        self.fine_tuning = False
        self.alpha = 1

    def on_train_begin(self):
        assert hasattr(self.dataset, "increase_resolution"), \
            f"{type(self.dataset)} must have attribute increase_resolution"
        assert hasattr(self.generator, "increase_resolution"), \
            f"{type(self.generator)} must have attribute increase_resolution"

    def on_epoch_begin(self, e):
        if self.epoch_counter == self.epochs:
            self.epoch_counter = 0

            if self.fade_in:
                self.epochs = int(self.epochs * self.increase_rate)
                self.fade_in = False
            else:
                self.dataset.increase_resolution()
                self.generator.increase_resolution()
                self.fade_in = True

        print("fading in") if self.fade_in and self.verbose else None

    def on_batch_begin(self, batch):
        alpha = 1

        if self.fade_in:
            e = self.epoch_counter
            alpha = (e + 1) * (batch + 1) / self.epochs / self.iterations

        self.trainer.d_kwarg["alpha"] = alpha
        self.trainer.g_kward["alpha"] = alpha

    def on_epoch_end(self, e, logs=None):
        self.epoch_counter += 1


class ModelCheckpoints(Callback):

    def __init__(self, directory, save_discriminator=False, save_optimizers=False, verbose=1):
        super().__init__()

        self.dir = directory
        self.save_discriminator = save_discriminator
        self.save_optimizers = save_optimizers
        self.verbose = verbose

    def on_epoch_end(self, e, logs=None):
        torch.save(self.generator.state_dict(), f"{self.dir}/{e}.generator")

        if self.save_discriminator:
            torch.save(self.discriminator.state_dict(), f"{self.dir}/{e}.discriminator")

        if self.save_optimizers:
            torch.save(self.trainer.d_opt.state_dict(), f"{self.dir}/{e}.d_opt")
            torch.save(self.trainer.g_opt.state_dict(), f"{self.dir}/{e}.g_opt")

        print("saving model to ", self.dir) if self.verbose else None
