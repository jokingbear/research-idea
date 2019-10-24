import torch
import torch.nn as nn
import torch.utils.data as data
import torch.autograd as ag

import numpy as np

from torch_modules.training.trainer import utils


class StandardGANTrainer:

    def __init__(self, discriminator, generator, optimizer, x_type=torch.float, x_device=None, r1_reg=10):
        self.discriminator = discriminator
        self.generator = generator
        self.optimizer = optimizer

        self.x_type = x_type
        self.x_device = x_device
        self.r1_reg = r1_reg or 0

        self.loss = nn.BCEWithLogitsLoss()

    def fit(self, train, epochs=1, batch_size=32, n_workers=0, callbacks=None):
        callbacks = callbacks or []

        [c.set_model_optimizer_trainer([self.discriminator, self.generator], self.optimizer, self) for c in callbacks]

        loader = data.DataLoader(train, batch_size, sampler=train if isinstance(train, data.RandomSampler) else None,
                                 shuffle=not isinstance(train, data.RandomSampler), num_workers=n_workers,
                                 pin_memory=True, drop_last=True)

        [c.on_train_begin() for c in callbacks]
        for e in range(epochs):
            print("Epoch: ", e + 1, "/", epochs)

            self.train_one_epoch(loader, callbacks)

            [c.on_epoch_end(e) for c in callbacks]
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, loader, callbacks):
        running_metrics = np.array([.0, .0])
        metric_names = ["d_loss", "g_loss"]

        with utils.get_pbar()(total=len(loader)) as pbar:
            for i, x in enumerate(loader):
                [c.on_batch_begin(i) for c in callbacks]
                x = utils.to_device(x, self.x_type, self.x_device)

                dloss = self.train_discriminator(*x)
                gloss = self.train_generator(*x)

                running_metrics += np.array([float(dloss), float(gloss)])
                logs = dict(zip(metric_names, running_metrics / (i + 1)))
                [c.on_batch_end(i, logs) for c in callbacks]

                pbar.set_postfix(logs, refresh=False)
                pbar.update()

    def train_discriminator(self, *x):
        d = self.discriminator.train()
        d.zero_grad()

        x = [ag.Variable(d, requires_grad=True) for d in x] if self.r1_reg > 0 else x
        real_score = d(*x)
        is_real = torch.ones(real_score.shape[0], 1, dtype=torch.float, device=real_score.device)
        loss1 = self.loss(real_score, is_real)

        if self.r1_reg > 0:
            grads = ag.grad(real_score.sum(), x, retain_graph=True, create_graph=True)
            grad_loss = torch.zeros([], dtype=torch.float, device=loss1.device)
            for g in grads:
                grad_loss = grad_loss + g.pow(2).sum(dim=[1, 2, 3])
            loss1 = loss1 + self.r1_reg * grad_loss.mean()
        loss1.backward()

        g = self.generator.train()
        with torch.no_grad():
            fakes = g()
            fakes = fakes if type(fakes) in {tuple, list} else [fakes]
        fake_score = d(*fakes)
        is_fake = torch.ones(fake_score.shape[0], 1, dtype=torch.float, device=fake_score.device)
        loss2 = self.loss(fake_score, is_fake)
        loss2.backward()

        loss = loss1 + loss2

        return loss.detach()

    def train_generator(self, *x):
        d = self.discriminator.train()
        g = self.generator.train()
        g.zero_grad()

        fakes = g()
        fakes = fakes if type(fakes) in {tuple, list} else [fakes]
        fake_scores = d(*fakes)
        is_real = torch.ones(fake_scores.shape[0], 1, dtype=torch.float, device=fake_scores.device)
        loss = self.loss(fake_scores, is_real)
        loss.backward()

        return loss.detach()
