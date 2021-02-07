import numpy as np
import pandas as pd

from plasma.training import data


df = pd.DataFrame(np.random.randint(0, 100_000, size=60_000).tolist())


def get_tensor(idx, *args, **kwargs):
    return np.random.randint(0, 256, size=[1, 28, 28]), np.array([0])


def train_valid():
    train = data.PandasDataset(df, get_tensor)
    loader = train.get_torch_loader(workers=8)

    return loader, loader
