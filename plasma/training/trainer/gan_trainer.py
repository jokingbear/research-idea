from itertools import count

import numpy as np
import torch
import torch.utils.data as data

from plasma.training.trainer import utils


class GANTrainer:

    def __init__(self, discriminator, generator_discriminator,
                 discriminator_optimizer, generator_optimizer,
                 loss, real_label=1, fake_label=0, regularizers=None,
                 metrics=None, device=None):
        self.discriminator = discriminator
        self.generator_discriminator = generator_discriminator

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.loss = loss
        self.real_label = real_label
        self.fake_label = fake_label
        self.regularizers = regularizers or []

        self.metrics = metrics or []
        self.device = device

        self.d_kwargs = {}
        self.g_kwargs = {}
        self.stop_training = False

    def fit(self, train, batch_size=32, generator_batch_size=None, callbacks=None, workers=0):
        callbacks = callbacks or []
        generator_batch_size = generator_batch_size or batch_size
        sampler = train.get_sampler() if hasattr(train, "get_sampler") else None
        loader = data.DataLoader(train, batch_size, sampler=sampler, shuffle=sampler is None,
                                 drop_last=True, num_workers=workers, pin_memory=True)

        self.g_kwargs["batch_size"] = generator_batch_size
        [c.set_trainer(self) for c in callbacks]
        [c.on_train_begin(loader=loader, generator_batch_size=generator_batch_size) for c in callbacks]
        for e in count(start=0):
            print("epoch ", e + 1)

            [c.on_epoch_begin(e) for c in callbacks]
            logs = self.train_one_epoch(loader, callbacks)
            [c.on_epoch_end(e, logs) for c in callbacks]
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, callbacks):
        n = len(train)
        metric_names = ["d_loss", "g_loss"] + [m.__name__ for m in self.metrics]
        running_metrics = np.zeros(len(metric_names) + 2)

        with utils.get_tqdm()(total=n) as pbar:
            for i, xs in enumerate(train):
                [c.on_batch_begin(i) for c in callbacks]
                xs = utils.to_device(xs, device=self.device, return_array=True)

                d_loss = self.train_discriminator(*xs)
                g_loss = self.train_generator()

                with torch.no_grad():
                    metrics = [float(m(self.generator_discriminator)) for m in self.metrics]
                    metrics = [float(d_loss), float(g_loss)] + metrics
                    running_metrics = running_metrics + metrics

                logs = dict(zip(metric_names, running_metrics / (i + 1)))
                [c.on_batch_end(i, logs) for c in callbacks]

                pbar.set_postfix(logs, refresh=False)
                pbar.update()

        return logs

    def train_discriminator(self, *x):
        self.discriminator.train().zero_grad()
        self.generator_optimizer.train().zero_grad()

        if len(self.regularizers) > 0:
            for a in x:
                a.requires_grad = True

        real_scores = self.discriminator(*x)
        real_labels = self.real_label * torch.ones(real_scores.shape, device=real_scores.device)
        real_loss = self.loss(real_scores, real_labels)

        if len(self.regularizers) > 0:
            for r in self.regularizers:
                real_loss = real_loss + r(x, real_scores)

        real_loss.backward()

        fake_scores = self.generator_discriminator(generator_grad=False, **self.g_kwargs)
        fake_labels = self.fake_label * torch.ones(fake_scores.shape, device=fake_scores.device)
        fake_loss = self.loss(fake_scores, fake_labels)
        fake_loss.backward()

        self.discriminator_optimizer.step()
        loss = fake_loss + real_loss

        return loss.detach()

    def train_generator(self):
        self.generator_discriminator.train().zero_grad()

        fake_scores = self.generator_discriminator(generator_grad=True, **self.g_kwargs)
        real_labels = self.real_label * torch.ones(fake_scores.shape, device=fake_scores.device)
        loss = self.loss(fake_scores, real_labels)
        loss.backward()

        self.generator_optimizer.step()

        return loss.detach()
