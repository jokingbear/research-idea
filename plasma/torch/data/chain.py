from .base import BaseDataset
from functools import cached_property


class ChainDataset(BaseDataset):

    def __init__(self, *datasets:BaseDataset):
        super().__init__()

        self.datasets = datasets
    
    def get_len(self):
        return sum([len(d) for d in self.datasets], 0)
    
    def get_item(self, idx):
        offset = 0
        for ds in self.datasets:
            if offset <= idx < offset + len(ds):
                return ds[idx - offset]
            offset += len(ds)

        raise IndexError(f'index {idx} is out of bound')

    @cached_property
    def offsets(self):
        offsets = [0]
        for d in self.datasets[:-1]:
            offsets.append(offsets[-1] + len(d))

        return offsets
