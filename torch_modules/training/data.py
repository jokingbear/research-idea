import torch

from abc import abstractmethod
from queue import Queue
from threading import Thread
from torch.utils import data


class StandardDataset(data.Dataset, data.RandomSampler):

    def __init__(self):
        super().__init__(self)

    def __len__(self):
        return self.get_len()

    def __getitem__(self, idx):
        return self.get_item(idx)

    def __iter__(self):
        self.sample()

        return super().__iter__()

    @abstractmethod
    def get_len(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    def sample(self):
        pass
