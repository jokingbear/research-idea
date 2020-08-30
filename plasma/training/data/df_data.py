import pandas as pd

from .base_class import StandardDataset


class PandasDataset(StandardDataset):

    def __init__(self, df: pd.DataFrame, mapper, imapper=None, **kwargs):
        """
        :param df: dataframe
        :param mapper: mapping function with signature idx, row -> tensors
        :param kwargs: additional argument to add to mapper
        """
        super().__init__()
        assert (mapper is not None) != (imapper is not None), "either mapper or imapper must be none"

        self.input_idx = imapper is not None
        self.df = df.copy()
        self.mapper = mapper
        self.imapper = imapper
        self.kwargs = kwargs

    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]

        if self.mapper is not None:
            return self.mapper(row, **self.kwargs)
        else:
            return self.imapper(idx, row, **self.kwargs)
