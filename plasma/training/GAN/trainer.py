from itertools import count

import numpy as np
import torch
from torch.utils.data import DataLoader

import plasma.training.utils as utils


class GANTrainer:

    def __init__(self, discriminator, generator, generator_discriminator,
                 discriminator_optimizer, generator_optimizer,
                 loss, metrics=None, dtype=torch.float, device=None):
        self.discriminator = discriminator
        self.generator = generator
        self.generator_discriminator = generator_discriminator

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.loss = loss
        self.metrics = metrics or []

        self.dtype = dtype
        self.device = device

        self.d_kwargs = {}
        self.g_kwargs = {}
        self.stop_training = False

    def fit(self, train, batch_size=32, callbacks=None, workers=0):
        callbacks = callbacks or []

        sampler = train.get_sampler() if hasattr(train, "get_sampler") else None
        loader = DataLoader(train, batch_size, shuffle=sampler is None, sampler=sampler,
                            num_workers=workers, pin_memory=True)

        [c.set_trainer(self) for c in callbacks]
        [c.on_train_begin(loader=loader) for c in callbacks]
        for e in count(start=0):
            print("epoch ", e + 1)

            [c.on_epoch_begin(e) for c in callbacks]
            logs = self.train_one_epoch(loader, callbacks)
            [c.on_epoch_end(e, logs) for c in callbacks]

            if self.stop_training:
                break
        [c.one_train_end() for c in callbacks]

    def train_one_epoch(self, loader, callbacks):
        n = len(loader)
        metric_names = ["d_loss", "g_loss"] + [m.__name__ for m in self.metrics]
        running_metrics = np.zeros(len(metric_names))

        with utils.get_tqdm()(total=n) as pbar:
            for i, reals in enumerate(loader):
                [c.on_batch_begin(i) for c in callbacks]
                d_loss = self.train_discriminator(*reals)
                g_loss = self.train_generator()

                metrics = self.get_metrics(d_loss, g_loss)
                running_metrics += metrics
                logs = dict(zip(metric_names, running_metrics / (i + 1)))

                pbar.set_postfix(logs, refresh=False)
                pbar.update()

        return logs

    def get_metrics(self, d_loss, g_loss):
        with torch.no_grad():
            self.generator.eval()
            metrics = [float(d_loss), float(g_loss)] + [m(self.generator, self.g_kwargs) for m in self.metrics]
            metrics = np.array(metrics)

        return metrics

    def train_discriminator(self, *reals):
        self.discriminator.train().zero_grad()
        self.generator_discriminator.train().zero_grad()

        if self.loss.requires_real_grads:
            for r in reals:
                r.requires_grad = True

        real_scores = self.discriminator(*reals, **self.d_kwargs)
        fake_scores, fakes = self.generator_discriminator(g_grad=False, g_kwargs=self.g_kwargs, d_kwargs=self.d_kwargs)
        loss = self.loss.discriminator_loss(reals, real_scores, fakes, fake_scores)

        loss.backward()
        self.discriminator_optimizer.step()

        return loss.detach()

    def train_generator(self):
        self.generator_discriminator.train().zero_grad()

        fake_scores, fakes = self.generator_discriminator(g_grad=True, **{**self.g_kwargs, **self.d_kwargs})
        loss = self.loss.generator_loss(fakes, fake_scores)

        loss.backward()
        self.generator_optimizer.step()

        return loss.detach()

# TODO: implement standard loss
# TODO: implement callbacks
