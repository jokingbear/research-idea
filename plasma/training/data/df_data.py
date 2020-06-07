import numpy as np
import pandas as pd

from .base_class import StandardDataset


class PandasDataset(StandardDataset):

    def __init__(self, df: pd.DataFrame, img_column, class_columns, img_reader=np.load, augmentations=None):
        super().__init__()

        assert isinstance(class_columns, list), "class_columns must be a list"

        self.df = df.copy().reset_index(drop=True)
        self.img_column = img_column
        self.class_columns = class_columns
        self.reader = img_reader
        self.augmentations = augmentations

    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]
        img = self.reader(row[self.img_column])

        if self.augmentations is not None:
            img = self.augmentations(image=img)["image"]

        if self.class_columns is not None:
            classes = row[self.class_columns].values

            return img, classes

        return img
