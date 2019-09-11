import numpy as np

from tqdm import tqdm


class Trainer:

    def __init__(self, model, optimizer, loss, metrics=None):
        self.model = model.train()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []

    def fit(self, train, test=None, epochs=1, callbacks=None, pbar=None):
        callbacks = callbacks or []
        pbar = pbar or tqdm

        [c.set_model_optimizer(self.model, self.optimizer) for c in callbacks]
        [c.on_train_begin() for c in callbacks]

        for e in range(epochs):
            print(f"epochs: {e + 1}/{epochs}")

            [c.on_epoch_begin(e) for c in callbacks]

            logs = self.train_one_epoch(train, pbar, callbacks)

            logs = self.evaluate_one_epoch(test, pbar, logs) if test is not None else logs

            [c.on_epoch_end(e, logs) for c in callbacks]
            train.on_epoch_end()
            test.on_epoch_end() if test is not None else None

        [c.on_train_end() for c in callbacks]

    def train_one_epoch(self, train, pbar, callbacks):
        n = len(train)
        running_metrics = np.zeros(shape=len(callbacks) + 1)

        with pbar(total=n) as pbar:
            for i in range(n):
                X, y = train[i]
                [c.on_batch_begin(i) for c in callbacks]

                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.loss(y, y_pred)
                loss.backward()
                self.optimizer.step()

                current_metrics = self.get_metrics(float(loss), y, y_pred)
                running_metrics = self.get_running_metrics(i, running_metrics, current_metrics)

                logs, logs_msg = self.get_logs(running_metrics)

                [c.on_batch_end(i, logs) for c in callbacks]
                pbar.set_postfix_str(logs_msg)
                pbar.update(1)

        return logs

    def evaluate_one_epoch(self, test, pbar, logs):
        running_metrics = np.zeros(len(self.metrics) + 1)
        n = len(test)

        val_logs = {}
        logs_msg = None

        for i in range(n):
            X, y = test[i]
            y_pred = self.model(X)
            loss = self.loss(y, y_pred)

            metrics = self.get_metrics(loss, y, y_pred)
            running_metrics = self.get_running_metrics(i, running_metrics, metrics)
            val_logs, logs_msg = self.get_logs(running_metrics, prefix="val")
            pbar.update(1)

        pbar.set_postfix_str(logs_msg)

        return {**logs, **val_logs}

    def get_metrics(self, loss, y, y_pred):
        metrics = [m(y, y_pred) for m in self.metrics]
        metrics = [loss] + metrics

        return np.array(metrics)

    def get_running_metrics(self, i, running_metrics, metrics):
        i1 = i + 1
        new_running_metrics = metrics / i1 + i * running_metrics / i1

        return new_running_metrics

    def get_logs(self, running_metrics, prefix=None):
        prefix = prefix or "training"
        names = ["loss"] + [m.name for m in self.metrics]
        logs = {f"{prefix}_{m}": m_val for m, m_val in zip(names, running_metrics)}
        msg = ", ".join([f"{name}: {logs[name]:.4f}" for name in logs.keys()])

        return logs, msg

    def get_final_model(self):
        return self.model.eval()