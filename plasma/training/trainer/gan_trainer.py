import numpy as np
import torch
import torch.autograd as ag
import torch.utils.data as data

from plasma.training.trainer import utils


class GANTrainer:

    def __init__(self, discriminator, generator, d_optimizer, g_optimizer, loss, metrics=None,
                 r1=None, real_label=1, fake_label=0, x_device=None):
        self.discriminator = discriminator
        self.generator = generator

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.loss = loss
        self.metrics = metrics or []
        self.r1 = r1 or 0

        self.real_label = real_label
        self.fake_label = fake_label
        self.x_device = x_device

        self.d_kwargs = {}
        self.g_kwargs = {}
        self.stop_training = False

    def fit(self, train, epochs=100, batch_size=32, g_batch_size=None, workers=0, callbacks=None):
        sampler = train.get_sampler() if hasattr(train, "get_sampler") else None
        loader = data.DataLoader(train, batch_size, sampler=sampler, shuffle=sampler is None, num_workers=workers,
                                 pin_memory=True, drop_last=True)
        self.g_kwargs["batch_size"] = g_batch_size or batch_size
        callbacks = callbacks or []

        [c.set_trainer_dataset(self, train) for c in callbacks]
        [c.on_train_begin() for c in callbacks]

        for e in range(epochs):
            print("epoch ", e + 1, "/", epochs)
            [c.on_epoch_begin(e) for c in callbacks]

            logs = self.train_one_epoch(loader, callbacks)

            with torch.no_grad():
                [c.on_epoch_end(e, logs) for c in callbacks]
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, loader, callbacks):
        logs = {}
        n = len(loader)
        metrics_names = ["d_loss", "g_loss"] + [m.__name__ for m in self.metrics]
        running_metrics = np.zeros(len(metrics_names))

        with utils.get_tqdm()(total=n) as pbar:
            for i, x in enumerate(loader):
                [c.on_batch_begin(i) for c in callbacks]
                x = utils.to_device(x, device=self.x_device)

                d_loss = self.train_discriminator(*x)
                g_loss = self.train_generator(*x)

                with torch.no_grad():
                    metrics = self.get_metrics(d_loss, g_loss)
                    running_metrics += metrics

                    logs = dict(zip(metrics_names, running_metrics / (i + 1)))
                    [c.on_batch_end(i, logs) for c in callbacks]

                pbar.set_postfix(logs, refresh=False)
                pbar.update()

        return logs

    def train_discriminator(self, *x):
        self.discriminator.train().zero_grad()
        self.generator.train().zero_grad()

        x = [ag.Variable(a, requires_grad=True) for a in x] if self.r1 else x
        real_score = self.discriminator(*x, **self.d_kwargs)
        real_label = self.real_label * torch.ones(real_score.shape, dtype=real_score.dtype, device=real_score.device)
        loss1 = self.loss(real_score, real_label)

        if self.r1 > 0:
            grads = ag.grad(real_score.sum(), x, retain_graph=True, create_graph=True, only_inputs=True)

            grad_loss = 0
            for g in grads:
                grad_loss = grad_loss + g.pow(2).sum(dim=[2, 3])
            loss1 = loss1 + grad_loss.mean() * self.r1 / 2

        loss1.backward()

        with torch.no_grad():
            fakes = self.generator(**self.g_kwargs)
            fakes = fakes if not torch.is_tensor(fakes) else [fakes]
        fake_score = self.discriminator(*fakes, **self.d_kwargs)
        fake_label = self.fake_label * torch.ones(fake_score.shape, dtype=fake_score.dtype, device=fake_score.device)
        loss2 = self.loss(fake_score, fake_label)
        loss2.backward()

        self.d_optimizer.step()
        loss = loss1 + loss2

        return loss.detach()

    def train_generator(self, *_):
        self.discriminator.train().zero_grad()
        self.generator.train().zero_grad()

        fake = self.generator(**self.g_kwargs)
        fake_score = self.discriminator(fake, **self.d_kwargs)
        real_label = self.real_label * torch.ones(fake_score.shape, dtype=fake_score.dtype, device=fake_score.device)
        loss = self.loss(fake_score, real_label)
        loss.backward()

        self.g_optimizer.step()

        return loss.detach()

    def get_metrics(self, d_loss, g_loss):
        self.generator.eval()
        metrics = [float(m(self.generator)) for m in self.metrics]
        metrics = [float(d_loss), float(g_loss), *metrics]

        return np.array(metrics)
