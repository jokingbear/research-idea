from abc import abstractmethod

from torch.utils import data


class StandardDataset(data.Dataset):

    def __init__(self):
        super(StandardDataset, self).__init__()

        self.sampler = None

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

    def get_torch_loader(self, batch_size=32, workers=20, pin=True, drop_last=True):
        sampler = self.sampler or RandomSampler(self)

        return data.DataLoader(self, batch_size, sampler=sampler, num_workers=workers,
                               pin_memory=pin, drop_last=drop_last)


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
