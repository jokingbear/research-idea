import torch
import numpy as np

from torch_modules.training import StandardDataset
from torch.utils import data


class Data(StandardDataset):

    def get_len(self):
        return 10

    def get_item(self, idx):
        return idx

    def sample(self):
        print("sample")


d = Data()
loader = data.DataLoader(d, sampler=d.get_sampler(), batch_size=4)

for _ in loader:
    print(_)
