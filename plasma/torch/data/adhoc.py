import pandas as pd

from .base import BaseDataset


class AdhocData(BaseDataset):

    def __init__(self, arr, mapping, kwargs=None):
        super().__init__()

        self.source = arr
        self.mapping = mapping
        self.kwargs = kwargs or {}

    def get_len(self):
        return len(self.source)

    def get_item(self, idx):
        if isinstance(self.source, (pd.DataFrame, pd.Series)):
            item = self.source.iloc[idx]
        else:
            item = self.source[idx]

        return self.mapping(item, **self.kwargs)
