import pandas as pd

from .base import BaseDataset


class DynamicDataset(BaseDataset):

    def __init__(self, data) -> None:
        super().__init__()

        self._data = data
    
    def get_len(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        item = self.pick_element(idx)
        return self.get_item(item)
    
    def pick_element(self, idx):
        data = self._data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx]
        else:
            return data[idx]
