import torch
import torch.utils.data as dt

from abc import abstractmethod


class Sequence(dt.Dataset):

    def __init__(self, x_type=torch.FloatTensor, y_type=torch.LongTensor):
        self.x_type = x_type
        self.y_type = y_type

    def __getitem__(self, idx):
        xs, ys = self.get_item(idx)

        if xs is list:
            xs = [torch.from_numpy(x).type(self.x_type) for x in xs]
        else:
            xs = torch.from_numpy(xs).type(self.x_type)

        if ys is list:
            ys = [torch.from_numpy(y).type(self.y_type) for y in ys]
        else:
            ys = torch.from_numpy(ys).type(self.y_type)

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
