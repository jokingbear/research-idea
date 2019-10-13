import torch
import numpy as np

from torch import utils
from torch.utils.data import DataLoader, Dataset, SequentialSampler, BatchSampler, Sampler, RandomSampler
from torch_modules.training import data


class Test(Dataset, RandomSampler):

    def __init__(self):
        super().__init__(self)

    def __len__(self):
        return self.get_len()

    def __getitem__(self, item):
        return self.get_item(item)

    def __iter__(self):
        self.shuffle()

        return super().__iter__()

    def get_len(self):
        return 10

    def get_item(self, idx):
        return [idx, idx], [idx, idx]

    def shuffle(self):
        print("shuffle")


a = Test()
d = DataLoader(a, batch_size=3, sampler=a, drop_last=True, pin_memory=True)

b = [x for x in d]

x, y = b[0]