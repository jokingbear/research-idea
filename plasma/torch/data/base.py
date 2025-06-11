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

    def get_torch_loader(self, 
        batch_size=32, 
        workers=20, 
        sampler=None,
        pin=True, 
        collator=None,
        drop_last=True, 
        shuffle=True,
        rank=None, 
        num_replicas=None
    ) -> data.DataLoader:
        """
        Args:
            batch_size: batch size to load
            workers: number of workers
            sampler: sampler to sample data
            pin: whether to pin gpu memory
            drop_last: whether to drop remaining data that can't be fitted in a batch
            shuffle: whether to shuffle the data
            rank: distributed ranking
            num_replicas: number of distribution replica
        Returns: Iterator
        """

        if sampler is None:
            if rank is None:
                sampler = RandomSampler(self) if shuffle else SequentialSampler(self)
            else:
                assert num_replicas is not None, 'num replicas can\'t be None when use rank'
                sampler = DistributedSampler(self, rank=rank, num_replicas=num_replicas, shuffle=shuffle)
                batch_size = batch_size // num_replicas

        loader = data.DataLoader(self, batch_size, 
                                 sampler=sampler, 
                                 num_workers=workers, 
                                 pin_memory=pin, 
                                 drop_last=drop_last, 
                                 collate_fn=collator)

        return loader
