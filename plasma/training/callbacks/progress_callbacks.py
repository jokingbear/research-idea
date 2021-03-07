from rich.progress import Progress as ProgressBar, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from .base_class import Callback


class Progress(Callback):

    def __init__(self):
        super(Progress, self).__init__()

        self.epoch_id = None
        self.train_id = None
        self.valid_id = None
        self.progress = ProgressBar(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            "{task.fields}", refresh_per_second=1.25)

        self.train_total = 0
        self.valid_total = 0

    def on_train_begin(self, train_loader, valid_loader=None, **train_configs):
        self.progress.__enter__()

        self.train_total = len(train_loader)

        if valid_loader is not None:
            self.valid_total = len(valid_loader)
        else:
            self.valid_total = 0

    def on_training_batch_begin(self, epoch, step, inputs, targets):
        if self.train_id is None:
            self.train_id = self.progress.add_task("train", total=self.train_total)

    def on_training_batch_end(self, epoch, step, inputs, targets, caches, logs=None):
        self.progress.update(self.train_id, advance=1, **logs)

    def on_validation_batch_begin(self, epoch, step, inputs, targets):
        if self.valid_total > 0 and self.valid_id is None:
            self.valid_id = self.progress.add_task("eval", total=self.valid_total)

    def on_validation_batch_end(self, epoch, step, inputs, targets, caches):
        if self.valid_id is not None:
            self.progress.update(self.valid_id, advance=1)

    def on_epoch_end(self, epoch, logs=None):
        if self.valid_id is not None:
            self.progress.update(self.valid_id, **logs)
        else:
            self.progress.update(self.train_id, **logs)

        self.train_id = None
        self.valid_id = None

    def on_train_end(self):
        self.progress.__exit__(None, None, None)
