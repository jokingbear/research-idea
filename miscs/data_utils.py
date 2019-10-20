import numpy as np

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


def split_idc_class(idc, classes=None, test_size=0.1, seed=7, skip=None):
    split_class = StratifiedShuffleSplit if classes else ShuffleSplit
    splitter = split_class(test_size=test_size, random_state=seed)

    step = 0
    skip = skip or 0

    for train_idc, test_idc in splitter.split(idc, classes):
        if step == skip:
            return idc[train_idc], idc[test_idc]
        step += 1
