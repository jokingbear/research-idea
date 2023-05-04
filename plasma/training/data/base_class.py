from abc import abstractmethod

from torch.utils import data
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler


class BaseDataset(data.Dataset):

    def __len__(self):
        return self.get_len()

    def __getitem__(self, idx):
        return self.get_item(idx)

    @abstractmethod
    def get_len(self):
        return 0

    @abstractmethod
    def get_item(self, idx):
        return idx

    def get_torch_loader(self, batch_size=32, workers=20, sampler=None, pin=True, drop_last=True, shuffle=True, 
                         rank=None, num_replicas=None):
        if rank is None:
            sampler = sampler or RandomSampler(self) if shuffle else SequentialSampler(self)
        else:
            assert num_replicas is not None, 'num replicas can be None when use rank'
            sampler = DistributedSampler(self, rank=rank, num_replicas=num_replicas, shuffle=shuffle)
            batch_size = batch_size // num_replicas

        return data.DataLoader(self, batch_size, sampler=sampler, num_workers=workers,
                               pin_memory=pin, drop_last=drop_last)
