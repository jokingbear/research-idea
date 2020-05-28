import numpy as np
import pandas as pd

devices = 8


def shuffle_indices(batch_size):
    idc = np.arange(0, batch_size)
    batch_per_device = batch_size // devices

    step = np.random.randint(1, devices) * batch_per_device
    shuffled_idc = (idc + step) % batch_size
    inversed_idc = (idc - step + batch_size) % batch_size

    return shuffled_idc, inversed_idc


a = np.random.rand(256)
shuffled, inversed = shuffle_indices(256)
b = a[shuffled]
c = b[inversed]
pd.DataFrame(np.stack([a, b, c], axis=-1), columns=["original", "shuffle", "inverse"])
