import time

import torch

from plasma.training import data


class Check(data.Dataset):

    def get_len(self):
        return 129

    def get_item(self, idx):
        a = idx * torch.ones(32, 32)
        time.sleep(1)
        return a, idx
