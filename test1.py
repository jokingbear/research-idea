import numpy as np
import torch

from sklearn.datasets import load_digits
from torchvision import utils

a = torch.tensor(load_digits(return_X_y=False).images[:25, np.newaxis])
shuffle_idc = np.random.choice(a.shape[0], size=a.shape[0], replace=False)
mapping_idc = [(old, new) for new, old in enumerate(shuffle_idc)]
inverse_idc = [new for _, new in sorted(mapping_idc, key=lambda kv: kv[0])]

b = a[shuffle_idc]
c = b[inverse_idc]

utils.save_image(a, "original.png", nrow=5)
utils.save_image(b, "shuffle.png", nrow=5)
utils.save_image(c, "inverse.png", nrow=5)
