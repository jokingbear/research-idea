from abc import abstractmethod

from torch.utils import data
from .loader import PrefetchLoader


class StandardDataset(data.Dataset):

    def __len__(self):
        return self.get_len()

    def __getitem__(self, idx):
        return self.get_item(idx)

    @abstractmethod
    def get_len(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    def reset(self):
        pass

    def get_sampler(self):
        return RandomSampler(self)

    def get_torch_loader(self, batch_size=32, workers=8, pin=True, drop_last=True):
        return data.DataLoader(self, batch_size, sampler=self.get_sampler(), num_workers=workers,
                               pin_memory=pin, drop_last=drop_last)

    def get_prefetch_loader(self, batch_size, shuffle=True, drop_last=True, prefetch_batch=2, pool=8):
        return PrefetchLoader(self, batch_size, shuffle, drop_last, prefetch_batch, pool)


class RandomSampler(data.RandomSampler):

    def __init__(self, dataset):
        super().__init__(dataset, replacement=False)

        self.dataset = dataset

    def __iter__(self):
        self.dataset.reset()

        return super().__iter__()


class SequentialSampler(data.SequentialSampler):

    def __init__(self, dataset):
        super().__init__(dataset)

        self.dataset = dataset

    def __iter__(self):
        self.dataset.reset()

        return super().__iter__()
