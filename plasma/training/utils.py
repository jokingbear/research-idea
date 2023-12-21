import os

import torch

from .data.adhoc_data import AdhocData


def eval_modules(*modules):
    """
    turn module into evaluation mode, with torch no grad
    :param modules: array of modules
    :return: torch.no_grad()
    """
    [m.eval() for m in modules]

    return torch.no_grad()


def get_loader(arr, mapper, batch_size=32, pin_memory=True, workers=None, progress=True, **kwargs):
    """
    get loader from array or dataframe
    :param arr: array to iter
    :param mapper: how to map array element, signature: elem -> obj
    :param batch_size: the batch size of the loader
    :param pin_memory: whether the loader should pin memory for fast transfer
    :param workers: number of workers to run in parallel
    :param progress: whether to show progress bar
    :return: pytorch loader
    """
    workers = workers or batch_size // 2
    dataset = AdhocData(arr, mapper, kwargs)
    loader = dataset.get_torch_loader(batch_size, workers, pin=pin_memory, drop_last=False,
                                      shuffle=False, progress=progress)
    return loader


def set_devices(*device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device_ids])
