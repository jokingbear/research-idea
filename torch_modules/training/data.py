import torch

from abc import abstractmethod
from queue import Queue
from threading import Thread


class Sequence:

    def __init__(self, x_type=torch.float, y_type=torch.long, x_device=None, y_device=None, prefetch=0):
        self.x_type = x_type
        self.y_type = y_type

        self.x_device = x_device
        self.y_device = y_device

        self.workers = prefetch + 1 if prefetch > 0 else 0
        self.queue = Queue(maxsize=prefetch + 1) if prefetch > 0 else None

    def __getitem__(self, idx):
        if self.workers == 0:
            xs, ys = self.copy_to_device(*self.get_item(idx))
        else:
            xs, ys = self.queue.get(block=True, timeout=None)
            self.queue.task_done()

        return xs, ys

    def __len__(self):
        return self.get_len()

    def __iter__(self):
        self.shuffle()
        workers = [Thread(target=self.fetch, args=(i, self.workers)) for i in range(self.workers)]
        [w.start() for w in workers]

        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def get_len(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    @abstractmethod
    def shuffle(self):
        pass

    def fetch(self, start, step):
        for i in range(start, len(self), step):
            xs, ys = self.get_item(i)
            xs, ys = self.copy_to_device(xs, ys)

            self.queue.put((xs, ys), block=True, timeout=None)

    def copy_to_device(self, xs, ys):
        if type(xs) is list:
            xs = [to_device(x, self.x_type, self.x_device) for x in xs]
        else:
            xs = to_device(xs, self.x_type, self.x_device)

        if type(ys) is list:
            ys = [to_device(y, self.y_type, self.y_device) for y in ys]
        else:
            ys = to_device(ys, self.y_type, self.y_device)

        return xs, ys


class GanSequence:

    def __init__(self, x_type=torch.float, x_device=None, prefetch=0):
        self.x_type = x_type
        self.x_device = x_device

        self.workers = prefetch + 1 if prefetch > 0 else 0
        self.queue = Queue(maxsize=self.workers) if self.workers > 0 else None
        self._workers = None

    def __len__(self):
        return self.get_len()

    def __getitem__(self, idx):
        if self.workers == 0:
            xs = self.copy_to_device(self.get_item(idx))
        else:
            xs = self.queue.get(block=True, timeout=None)

        return xs

    def __iter__(self):
        self.shuffle()
        workers = [Thread(target=self.fetch, args=(i, self.workers)) for i in range(self.workers)]
        [w.start() for w in workers]
        self._workers = workers

        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def get_len(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    @abstractmethod
    def shuffle(self):
        pass

    def fetch(self, start, step):
        for i in range(start, len(self), step):
            xs = self.get_item(i)
            xs = self.copy_to_device(xs)

            self.queue.put(xs, block=True, timeout=None)

    def copy_to_device(self, xs):
        if type(xs) is list:
            xs = [to_device(x, self.x_type, self.x_device) for x in xs]
        else:
            xs = to_device(xs, self.x_type, self.x_device)

        return xs


def to_device(x, dtype, device):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=dtype, device=device)
    elif x.device != device:
        x = x.to(device)

    return x
