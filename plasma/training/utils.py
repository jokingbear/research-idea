import multiprocessing as mp
from multiprocessing.dummy import Pool
import os

import numpy as np
import torch
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from .data.adhoc_data import AdhocData


notebook = False


def get_progress(iterable=None, total=None, desc=None):
    """
    get progress bar
    :param iterable: target to be iterated
    :param total: total length of the progress bar
    :param desc: description of the progress bar
    :return: progress bar
    """

    if notebook:
        return tqdm_nb(iterable=iterable, total=total, desc=desc)

    return tqdm(iterable=iterable, total=total, desc=desc)


def eval_modules(*modules):
    """
    turn module into evaluation mode, with torch no grad
    :param modules: array of modules
    :return: torch.no_grad()
    """
    [m.eval() for m in modules]

    return torch.no_grad()


def parallel_iterate(arr, iter_func, workers=8, batch_size=1, **kwargs):
    """
    parallel iterate array
    :param arr: array to be iterated
    :param iter_func: function to be called for each data, signature (idx, arg) or arg
    :param workers: number of worker to run
    :param batch_size: chunk size
    :param use_index: whether to add index to each call of iter func
    :return list of result if not all is None
    """
    n = len(arr)
    n_chunk = n // batch_size
    if n_chunk * batch_size != n:
        n_chunk += 1

    with mp.Pool(workers) as p:
        runners = p.imap(iter_func, arr, batch_size)
        return [r for r in get_progress(runners, total=len(arr))]


def get_loader(arr, mapper=None, imapper=None, batch_size=32, pin_memory=True, workers=None, **kwargs):
    """
    get loader from array or dataframe
    :param arr: array to iter
    :param mapper: how to map array element, signature: elem -> obj
    :param imapper: how to map array element, signature: (idx, elem) -> obj
    :param batch_size: the batch size of the loader
    :param pin_memory: whether the loader should pin memory for fast transfer
    :param workers: number of workers to run in parallel
    :return: pytorch loader
    """
    workers = workers or batch_size // 2
    dataset = AdhocData(arr, mapper, imapper, kwargs)
    loader = dataset.get_torch_loader(batch_size, workers, pin=pin_memory, drop_last=False, shuffle=False)
    return loader


def set_devices(*device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device_ids])
