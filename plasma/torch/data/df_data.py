import pandas as pd

from .base_class import BaseDataset


class PandasDataset(BaseDataset):

    def __init__(self, df: pd.DataFrame, mapper, **kwargs):
        """
        :param df: dataframe
        :param mapper: mapping function with signature idx, row -> tensors
        :param kwargs: additional argument to add to mapper
        """
        super().__init__()

        self.df = df.copy()
        self.mapper = mapper
        self.kwargs = kwargs

    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]

        return self.mapper(row, **self.kwargs)
