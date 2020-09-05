from .sgd_gc import SGD_GC
from torch.optim import Adam, SGD


__mapping__ = {
    "sgd_gc": SGD_GC,
    "sgd gc": SGD_GC,
    "sgd": SGD,
    "adam": Adam
}
