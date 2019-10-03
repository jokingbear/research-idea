import torch
import imageio as io
import numpy as np

from abc import abstractmethod
from miscs import data_utils as utils


class Sequence:

    def __init__(self, x_type=torch.float, y_type=torch.long, x_gpu=None, y_gpu=None):
        self.x_type = x_type
        self.y_type = y_type
        self.x_gpu = x_gpu
        self.y_gpu = y_gpu

    def __getitem__(self, idx):
        xs, ys = self.get_item(idx)

        if type(xs) is list:
            xs = [torch.tensor(x, dtype=self.x_type, device=self.x_gpu) for x in xs]
        else:
            xs = torch.tensor(xs, dtype=self.x_type, device=self.x_gpu)

        if type(ys) is list:
            ys = [torch.tensor(y, dtype=self.y_type, device=self.y_gpu) for y in ys]
        else:
            ys = torch.tensor(ys, dtype=self.y_type, device=self.y_gpu)

        return xs, ys

    def __len__(self):
        return self.get_len()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def get_len(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    @abstractmethod
    def shuffle(self):
        pass


class BinarySegmentationSequence(Sequence):

    def __init__(self, img0s, img1s, normalize_fn, batch_size=1, train=True, **kwargs):
        super().__init__(**kwargs)

        self.img0s = img0s
        self.img1s = img1s
        self.normalize_fn = normalize_fn
        self.batch_size = batch_size
        self.train = train

        self.sample_img0s = None
        self.sample_img1s = None

    def get_len(self):
        return max(len(self.img0s), len(self.img1s)) // self.batch_size

    def get_item(self, idx):
        b = self.batch_size

        img0s, msk0s = zip(*[self.normalize_fn(img0, self.train) for img0 in self.img0s[idx*b:(idx + 1)*b]])
        img1s, msk1s = zip(*[self.normalize_fn(img1, self.train) for img1 in self.img1s[idx*b:(idx + 1)*b]])

        imgs = np.stack(img0s + img1s)
        msks = np.stack(msk0s + msk1s)

        return imgs[:, np.newaxis, ...], msks[:, np.newaxis, ...]

    def shuffle(self):
        n = max(len(self.img1s), len(self.img0s))

        img0s = utils.sample(self.img0s, n)
        img1s = utils.sample(self.img1s, n)

        self.sample_img0s = img0s
        self.sample_img1s = img1s
