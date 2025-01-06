import pandas as pd

from .base import BaseDataset
from abc import abstractmethod


class DynamicDataset(BaseDataset):

    def __init__(self, data) -> None:
        super().__init__()

        self._data = data
    
    def get_len(self):
        return len(self._data)
    
    def get_item(self, idx):
        item = self.pick_item(idx)
        return self.process_item(item)
    
    def pick_item(self, idx):
        data = self._data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx]
        else:
            return data[idx]

    @abstractmethod
    def process_item(self, item):
        pass
