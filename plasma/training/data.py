from abc import abstractmethod

from torch.utils import data


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

    def sample(self):
        pass

    def get_sampler(self):
        return CustomSampler(self)


class CustomSampler(data.RandomSampler):

    def __init__(self, dataset):
        super().__init__(dataset, replacement=False)

        self.dataset = dataset

    def __iter__(self):
        self.dataset.sample()

        return super().__iter__()
