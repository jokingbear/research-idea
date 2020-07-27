import pandas as pd

from .base_class import StandardDataset as Dataset, RandomSampler, SequentialSampler
from .df_data import PandasDataset


def get_pandas_dataset(df: pd.DataFrame, mapper, **kwargs):
    """
    create dataset from pandas dataframe
    :param df: pandas dataframe
    :param mapper: mapping from row to numpy array, mapping signature: (idx, **row) -> numpy arrays
    :param kwargs: additional parameter for mapper function
    :return: plasma Dataset
    """
    df = df.copy()

    class Data(Dataset):

        def get_len(self):
            return len(df)

        def get_item(self, idx):
            row = df.iloc[idx]

            return mapper(idx, row, **kwargs)

    return Data()
