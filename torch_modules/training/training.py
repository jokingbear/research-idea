import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm


class Trainer:

    def __init__(self, model, optimizer, loss, metrics=None):
        self.model = model.train()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.train_mode = True

    def fit(self, train, test=None, epochs=1, callbacks=None, pbar=None):
        callbacks = callbacks or []
        pbar = pbar or tqdm

        [c.set_model_optimizer_trainer(self.model, self.optimizer, self) for c in callbacks]
        [c.on_train_begin() for c in callbacks]

        for e in range(epochs):
            print(f"epochs: {e + 1}/{epochs}")

            [c.on_epoch_begin(e) for c in callbacks]

            train_logs = self.train_one_epoch(train, pbar, callbacks)

            val_logs = self.evaluate_one_epoch(test, pbar) if test is not None else {}

            logs = {**train_logs, **val_logs}

            [c.on_epoch_end(e, logs) for c in callbacks]
            train.shuffle()
            test.shuffle() if test is not None else None

            if not self.train_mode:
                break

        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, pbar=None, callbacks=None):
        pbar = pbar or tqdm
        n = len(train)
        callbacks = callbacks or []
        metrics_names = ["loss"] + [m.__name__ for m in self.metrics]
        running_metrics = np.zeros(shape=len(self.metrics) + 1)

        with pbar(total=n, desc="train") as pbar:
            for i, (x, y) in enumerate(train):
                [c.on_batch_begin(i) for c in callbacks]

                loss, y_pred = self.train_one_batch(x, y)

                with torch.no_grad():
                    current_metrics = self.get_metrics(loss, y, y_pred)
                    running_metrics += current_metrics

                    logs = dict(zip(metrics_names, running_metrics / (i + 1)))

                    [c.on_batch_end(i, logs) for c in callbacks]

                pbar.set_postfix(logs, refresh=False)
                pbar.update(1)

        return logs

    def train_one_batch(self, x, y):
        self.model.zero_grad()
        y_pred = self.model(x)
        loss = self.loss(y, y_pred)
        loss.backward()
        self.optimizer.step()

        return loss.detach(), y_pred.detach()

    def evaluate_one_epoch(self, test, pbar=None):
        model = self.model.eval()

        pbar = pbar or tqdm
        n = len(test)
        metrics_names = ["val_loss"] + ["val_" + m.__name__ for m in self.metrics]
        running_metrics = np.zeros(len(self.metrics) + 1)

        with pbar(total=n, desc="evaluate") as pbar, torch.no_grad():
            for i, (x, y) in enumerate(test):
                y_pred = model(x)
                loss = self.loss(y, y_pred)

                metrics = self.get_metrics(loss, y, y_pred)
                running_metrics += metrics
                pbar.update(1)

            val_logs = dict(zip(metrics_names, running_metrics / n))
            pbar.set_postfix(val_logs)

        return val_logs

    def get_metrics(self, loss, y, y_pred):
        metrics = [float(m(y, y_pred)) for m in self.metrics]
        metrics = [float(loss)] + metrics

        return np.array(metrics)


class GANTrainer:

    def __init__(self, discriminator, generator, d_optimizer, g_optimizer, loss, metrics=None):
        self.discriminator = discriminator
        self.generator = generator
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.train_mode = True

    def fit(self, train, epochs=1, callbacks=None, pbar=tqdm):
        callbacks = callbacks or []
        pbar = pbar or tqdm

        [c.set_model_optimizer_trainer([self.discriminator, self.generator], [self.d_optimizer, self.g_optimizer], self)
         for c in callbacks]

        [c.on_train_begin() for c in callbacks]

        for e in range(epochs):
            print(f"{e + 1}/{epochs}")

            [c.on_epoch_begin(e) for c in callbacks]

            train_logs = self.train_one_epoch(train, pbar, callbacks)

            [c.on_epoch_end(e, train_logs) for c in callbacks]

            train.shuffle()

            if not self.train_mode:
                break
        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, pbar, callbacks):
        n = len(train)
        metrics_names = ["real_loss", "fake_loss"] + [m.__name__ for m in self.metrics]
        running_metrics = np.zeros(shape=len(self.metrics) + 2)

        with pbar(total=n, desc="train") as pbar:
            for i, x in enumerate(train):
                [c.on_batch_begin(i) for c in callbacks]

                real_loss = self.train_discriminator(x)
                fake_loss = self.train_generator(x)

                with torch.no_grad():
                    current_metrics = self.get_metrics(real_loss, fake_loss)
                    running_metrics += current_metrics

                    logs = dict(zip(metrics_names, running_metrics / (i + 1)))
                    [c.on_batch_end(i, logs) for c in callbacks]

                pbar.set_postfix(logs, refresh=False)
                pbar.update(1)

        return logs

    def train_discriminator(self, x):
        d = self.discriminator.train()
        g = self.generator.train()

        d.zero_grad()
        real_labels = torch.ones(x.shape[0], dtype=torch.float, device=x.device)
        real_loss = self.loss(real_labels, d(x))
        real_loss.backward()
        self.d_optimizer.step()

        d.zero_grad()
        with torch.no_grad():
            fakes = g()
        fake_labels = torch.zeros(fakes.shape[0], dtype=torch.float, device=fakes.device)
        fake_loss = self.loss(fake_labels, d(fakes))
        fake_loss.backward()
        self.d_optimizer.step()

        return (real_loss + fake_loss).detach()

    def train_generator(self, x):
        g = self.generator.train()
        d = self.discriminator.train()

        g.zero_grad()
        fakes = g()
        score = d(fakes)
        labels = torch.ones(score.shape[0], dtype=torch.float, device=score.device)
        loss = self.loss(labels, score)
        loss.backward()
        self.g_optimizer.step()

        return loss.detach()

    def get_metrics(self, real_loss, fake_loss):
        g = self.generator.eval()
        metrics = [float(m(g)) for m in self.metrics]
        metrics = [float(real_loss), float(fake_loss)] + metrics

        return np.array(metrics)
