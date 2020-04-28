import numpy as np

from plasma.training import utils


a = np.arange(1, 100)

loader = utils.get_batch_iterator(a)
