import numpy as np
import os
import pandas as pd

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def standardize_kernel_stride_dilation(dimensions, check_type, value):
    if type(value) is int:
        return (value,) * dimensions
    elif (type(value) is list or type(value) is tuple) and len(value) == dimensions:
        return tuple(value)
    else:
        raise TypeError(f"{value} is not a valid {check_type}")


def up_sample(data, n_sample):
    n = len(data)

    n_repeat = n_sample // n
    n_add = n_sample % n

    repeats = [np.random.choice(data, size=n, replace=False) for _ in range(n_repeat)]
    adds = [np.random.choice(data, size=n_add, replace=False)]

    return np.concatenate(repeats + adds)


def normalize_img(img):
    return img / 127.5 - 1


def split_file(files, test_size=0.1, seed=7, skip=None):
    splitter = ShuffleSplit(test_size=test_size, random_state=seed)
    files = np.array(files)

    pick = skip or 0

    skip = 0
    for train_idc, test_idc in splitter.split(files):
        if skip == pick:
            train = files[train_idc]
            test = files[test_idc]

            return train, test

        skip += 1


def split_df(df, column="class", test_size=0.1, seed=7, skip=None):
    splitter = StratifiedShuffleSplit(test_size=test_size, random_state=seed)

    pick = skip or 0

    skip = 0
    for train_idc, test_idc in splitter.split(df, df[column]):
        if skip == pick:
            train = df.iloc[train_idc]
            test = df.iloc[test_idc]

            return train, test

        skip += 1


def make_df_from_folder(path, class_label):
    files = [f"{path}/{f}" for f in os.listdir(path)]

    df = pd.DataFrame(files, columns=["path"])

    if class_label is not None:
        df["class"] = class_label

    return df
