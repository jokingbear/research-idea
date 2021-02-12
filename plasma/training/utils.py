import multiprocessing as mp
import os

import torch
import torch.utils.data as data
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

on_notebook = False


def get_tqdm(iterable=None, total=None, desc=None):
    """
    get tqdm progress bar
    :param iterable: target to be iterated
    :param total: total length of the progress bar
    :param desc: description of the progress bar
    :return: tqdm progress bar
    """
    if on_notebook:
        return tqdm_nb(iterable=iterable, total=total, desc=desc)
    else:
        return tqdm(iterable=iterable, total=total, desc=desc)


def eval_modules(*modules):
    """
    turn module into evaluation mode, with torch no grad
    :param modules: array of modules
    :return: torch.no_grad()
    """
    [m.eval() for m in modules]

    return torch.no_grad()


def parallel_iterate(arr, iter_func, workers=32, use_index=False, **kwargs):
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

    results = [j.get() for j in get_tqdm(jobs)]
    pool.close()
    pool.join()

    if not all([r is None for r in results]):
        return results


def get_loader(arr, mapper=None, imapper=None, batch_size=32, pin_memory=True, workers=20, **kwargs):
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
    n = len(arr)
    workers = workers or batch_size // 2

    class Data(data.Dataset):

        def __len__(self):
            return n

        def __getitem__(self, idx):
            item = arr[idx]

            if mapper is not None:
                return mapper(item, **kwargs)
            elif imapper is not None:
                return mapper(idx, item, **kwargs)

            return item

    return data.DataLoader(Data(), batch_size, shuffle=False, drop_last=False,
                           pin_memory=pin_memory, num_workers=workers)


def set_devices(*device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device_ids])
