import numpy as np
import pandas as pd
import cv2

from .base_class import StandardDataset


class PandasDataset(StandardDataset):

    def __init__(self, df: pd.DataFrame, path_column, class_columns, prefix=None, img_reader=cv2.imread,
                 cast_label=np.uint8, augmentations=None):
        """
        Pandas dataset for classification
        :param df: source dataframe
        :param path_column: columns containing image path
        :param class_columns: columns containing labels
        :param img_reader: how to read image
        :param cast_label: cast label to type
        :param augmentations: additional albumentations augmentations
        """
        super().__init__()

        assert isinstance(class_columns, list), "class_columns must be a list"

        self.df = df.copy().reset_index(drop=True)
        self.img_column = path_column
        self.class_columns = class_columns
        self.prefix = prefix or ""
        self.reader = img_reader
        self.cast_label = cast_label
        self.augmentations = augmentations

    def get_len(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]
        path = row[self.img_column]
        path = self.prefix + path
        img = self.reader(path)

        if self.augmentations is not None:
            img = self.augmentations(image=img)["image"]

        if len(img.shape) == 2:
            img = img[np.newaxis]
        else:
            img = img.transpose([2, 0, 1])

        if self.class_columns is not None:
            classes = row[self.class_columns].values.astype(self.cast_label)

            return img, classes

        return img
