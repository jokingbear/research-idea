import pandas as pd

from .base import BaseDataset, abstractmethod


class DynamicDataset(BaseDataset):

    def __init__(self, data) -> None:
        super().__init__()

        self._data = data
    
    def get_len(self):
        return len(self._data)
    
    def get_item(self, idx):
        element = self.get_element(idx)
        return self.map_element(element)
    
    def get_element(self, idx):
        data = self._data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx]
        else:
            return data[idx]

    @abstractmethod    
    def map_element(self, element):
        pass
