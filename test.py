import numpy as np

from plasma.training.data.volumes import preprocess


a = np.array([0, 1, 2, 3, 4])
b = np.stack([a, a + 5, a + 10, a + 15, a + 20], axis=1)
c1 = preprocess.crop_or_pad(b, 1, 2)
c2 = preprocess.crop_or_pad(c1, 0, 2)

c3 = preprocess.crop_or_pad(c2, 0, 5)
c4 = preprocess.crop_or_pad(c3, 1, 5)
