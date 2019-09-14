import torch
import torch.utils.data as dt

from abc import abstractmethod


class Sequence(dt.Dataset):

    def __init__(self, x_type=torch.float, y_type=torch.long, x_gpu=None, y_gpu=None):
        self.x_type = x_type
        self.y_type = y_type
        self.x_gpu = x_gpu
        self.y_gpu = y_gpu

    def __getitem__(self, idx):
        xs, ys = self.get_item(idx)

        if type(xs) is list:
            xs = [_init_tensor_from_numpy(x, self.x_type, self.x_gpu) for x in xs]
        else:
            xs = _init_tensor_from_numpy(xs, self.x_type, self.x_gpu)

        if type(ys) is list:
            ys = [_init_tensor_from_numpy(y, self.y_type, self.y_gpu) for y in ys]
        else:
            ys = _init_tensor_from_numpy(ys, self.y_type, self.y_gpu)

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


def _init_tensor_from_numpy(numpy_val, dtype, gpu):
    return torch.tensor(numpy_val, dtype=dtype, device=gpu)
