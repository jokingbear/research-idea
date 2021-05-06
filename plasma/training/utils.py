import multiprocessing as mp
import os

import torch
from tqdm import tqdm
from .data.adhoc_data import AdhocData
from cleanlab.pruning import get_noise_indices


def get_progress(iterable=None, total=None, desc=None):
    """
    get progress bar
    :param iterable: target to be iterated
    :param total: total length of the progress bar
    :param desc: description of the progress bar
    :return: progress bar
    """
    return tqdm(iterable=iterable, total=total, desc=desc)


def eval_modules(*modules):
    """
    turn module into evaluation mode, with torch no grad
    :param modules: array of modules
    :return: torch.no_grad()
    """
    [m.eval() for m in modules]

    return torch.no_grad()


def parallel_iterate(arr, iter_func, workers=8, use_index=False, **kwargs):
    """
    parallel iterate array
    :param arr: array to be iterated
    :param iter_func: function to be called for each data, signature (idx, arg) or arg
    :param workers: number of worker to run
    :param use_index: whether to add index to each call of iter func
    :return list of result if not all is None
    """
    pool = mp.Pool(workers)
    jobs = [pool.apply_async(iter_func, args=(i, arg) if use_index else (arg,), kwds=kwargs)
            for i, arg in enumerate(arr)]

    results = [j.get() for j in get_progress(jobs)]
    pool.close()
    pool.join()

    if not all([r is None for r in results]):
        return results


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


def detect_outliers(preds, trues, multi_label=False):
    """
    detect outliers in data using prediction probability and ground truth
    Args:
        preds: prediction probability [N, class]
        trues: ground truth [N] or [N, class] in case of multi_class
        multi_label: whether to treat trues as multi-label or not
    Returns:
        indices of outliers
    """
    if multi_label:
        trues = [trues[:, i] for i in range(trues.shape[-1])]

    idc = get_noise_indices(trues, preds, multi_label=multi_label, sorted_index_method='normalized_margin')

    return idc
