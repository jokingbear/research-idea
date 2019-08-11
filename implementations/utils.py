import numpy as np


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
    n_add = n - n_sample % n

    repeats = [np.random.choice(data, size=n, replace=False) for _ in range(n_repeat)]
    adds = [np.random.choice(data, size=n_add, replace=False)]

    return np.concatenate(repeats + adds)


def normalize_img(img):
    return img / 127.5 - 1
