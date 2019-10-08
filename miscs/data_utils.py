import os
import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def sample(data, n_sample):
    n = len(data)

    if n >= n_sample:
        return np.random.choice(data, size=n_sample, replace=False)

    n_repeat = n_sample // n
    n_add = n_sample % n

    repeats = [np.random.choice(data, size=n, replace=False) for _ in range(n_repeat)]
    adds = [np.random.choice(data, size=n_add, replace=False)]

    return np.concatenate(repeats + adds, axis=0)


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


def shuffle_data(x, y=None):
    n = x.shape[0]

    idc = np.random.choice(n, size=n, replace=False)

    x = x[idc]
    y = y[idc] if y else None

    if y:
        return x, y

    return x