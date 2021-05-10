from .sgd_gc import SGD_GC, SGD_GCC, SGDW, SGDW_GCC
from torch.optim import Adam, SGD


__mapping__ = {
    "sgd": SGD,
    "adam": Adam,
    "sgd_gc": SGD_GC,
    'sgd_gcc': SGD_GCC,
    'sgdw': SGDW,
    'sgdw_gcc': SGDW_GCC,
}
