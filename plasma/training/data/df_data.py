import numpy as np
import pandas as pd

from .base_class import StandardDataset


class PandasDataset(StandardDataset):

    def __init__(self, df: pd.DataFrame, img_column, class_columns, img_reader=np.load, cast_label=np.uint8):
        super().__init__()

        assert isinstance(class_columns, list), "class_columns must be a list"

        self.df = df.copy().reset_index(drop=True)
        self.img_column = img_column
        self.class_columns = class_columns
        self.reader = img_reader
        self.cast_label = cast_label

    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]
        img = self.reader(row[self.img_column])

        if self.class_columns is not None:
            classes = row[self.class_columns].values.astype(self.cast_label)

            return img, classes

        return img
