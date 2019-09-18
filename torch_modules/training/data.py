import torch

from abc import abstractmethod


class Sequence:

    def __init__(self, x_type=torch.float, y_type=torch.long, x_gpu=None, y_gpu=None):
        self.x_type = x_type
        self.y_type = y_type
        self.x_gpu = x_gpu
        self.y_gpu = y_gpu

    def __getitem__(self, idx):
        xs, ys = self.get_item(idx)

        if type(xs) is list:
            xs = [torch.tensor(x, dtype=self.x_type, device=self.x_gpu) for x in xs]
        else:
            xs = torch.tensor(xs, dtype=self.x_type, device=self.x_gpu)

        if type(ys) is list:
            ys = [torch.tensor(y, dtype=self.y_type, device=self.y_gpu) for y in ys]
        else:
            ys = torch.tensor(ys, dtype=self.y_type, device=self.y_gpu)

        return xs, ys

    def __len__(self):
        return self.get_len()

    @abstractmethod
    def get_len(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

