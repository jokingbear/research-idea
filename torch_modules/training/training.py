import numpy as np
import torch

from tqdm import tqdm


def get_running_metrics(i, running_metrics, metrics):
    i1 = i + 1
    new_running_metrics = metrics / i1 + i * running_metrics / i1

    return new_running_metrics


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

            [c.on_epoch_end(e, {**train_logs, **val_logs}) for c in callbacks]
            train.on_epoch_end()
            test.on_epoch_end() if test is not None else None

            if not self.train_mode:
                break

        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, pbar=None, callbacks=None):
        n = len(train)
        running_metrics = np.zeros(shape=len(self.metrics) + 1)
        model = self.model.train()

        pbar = pbar or tqdm
        callbacks = callbacks or []

        with pbar(total=n, desc="training") as pbar:
            for i in range(n):
                x, y = train[i]
                [c.on_batch_begin(i) for c in callbacks]

                model.zero_grad()
                y_pred = model(x)
                loss = self.loss(y, y_pred)
                loss.backward()
                self.optimizer.step()

                current_metrics = self.get_metrics(loss, y, y_pred)
                running_metrics = get_running_metrics(i, running_metrics, current_metrics)

                logs, logs_msg = self.get_logs(running_metrics)

                [c.on_batch_end(i, logs) for c in callbacks]
                pbar.set_postfix_str(logs_msg)
                pbar.update(1)

        return logs

    def evaluate_one_epoch(self, test, pbar=None):
        running_metrics = np.zeros(len(self.metrics) + 1)
        n = len(test)

        model = self.model.eval()

        pbar = pbar or tqdm

        with pbar(total=n, desc="evaluate") as pbar, torch.no_grad():
            for i in range(n):
                x, y = test[i]
                y_pred = model(x)
                loss = self.loss(y, y_pred)

                metrics = self.get_metrics(loss, y, y_pred)
                running_metrics = get_running_metrics(i, running_metrics, metrics)
                pbar.update(1)

            val_logs, logs_msg = self.get_logs(running_metrics, prefix="val")
            pbar.set_postfix_str(logs_msg)

        return val_logs

    def get_metrics(self, loss, y, y_pred):
        metrics = [float(m(y, y_pred)) for m in self.metrics]
        metrics = [float(loss)] + metrics

        return np.array(metrics)

    def get_logs(self, running_metrics, prefix=None):
        prefix = f"{prefix}_" if prefix else ""
        names = ["loss"] + [m.__name__ for m in self.metrics]
        logs = {prefix + m: m_val for m, m_val in zip(names, running_metrics)}
        msg = ", ".join([f"{name}: {logs[name]:.4f}" for name in logs.keys()])

        return logs, msg

