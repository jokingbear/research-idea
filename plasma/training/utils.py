import pandas as pd
import multiprocessing as mp
import os

import torch
import torch.multiprocessing as torch_mp

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from .data.adhoc_data import AdhocData
from ..functional import auto_func
from queue import Empty


notebook = False


def get_progress(iterable=None, total=None, desc=None, leave=False):
    """
    get progress bar
    :param iterable: target to be iterated
    :param total: total length of the progress bar
    :param desc: description of the progress bar
    :return: progress bar
    """

    if notebook:
        return tqdm_nb(iterable=iterable, total=total, desc=desc, leave=leave)

    return tqdm(iterable=iterable, total=total, desc=desc, leave=leave)


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
    with mp.Pool(workers) as p:
        if isinstance(arr, zip):
            jobs = [p.apply_async(iter_func, args=(i, *arg) if use_index else arg, kwds=kwargs) for i, arg in enumerate(arr)]
        elif isinstance(arr, pd.DataFrame):
            jobs = [p.apply_async(iter_func, args=(i, row) if use_index else (row,), kwds=kwargs) for i, row in arr.iterrows()]
        else:
            jobs = [p.apply_async(iter_func, args=(i, arg) if use_index else (arg,), kwds=kwargs) for i, arg in enumerate(arr)]
        results = [j.get() for j in get_progress(jobs)]
        return results


def get_loader(arr, mapper, batch_size=32, pin_memory=True, workers=None, **kwargs):
    """
    get loader from array or dataframe
    :param arr: array to iter
    :param mapper: how to map array element, signature: elem -> obj
    :param batch_size: the batch size of the loader
    :param pin_memory: whether the loader should pin memory for fast transfer
    :param workers: number of workers to run in parallel
    :return: pytorch loader
    """
    workers = workers or batch_size // 2
    dataset = AdhocData(arr, mapper, kwargs)
    loader = dataset.get_torch_loader(batch_size, workers, pin=pin_memory, drop_last=False, shuffle=False)
    return loader


def set_devices(*device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device_ids])


def process_queue(running_context, process_func, nprocess=50, infinite_loop=True, task_name=None):
    """
    create a queue with nprocess to resolve that queue
    :param running_context: running function, receive queue as input
    :param process_func: function to process queue item
    :param nprocess: number of independent process
    :param infinite_loop: number of worker to run
    :param task_name: name for the running process
    """

    process_func = auto_func(process_func)
    def run_process(i, queue: mp.Queue, is_dones):
        condition = True

        while condition:
            try:
                item = queue.get()

                process_func(item)

                queue.task_done()
                condition &= infinite_loop & ~is_dones[i]
            except Empty:
                return

    with mp.Manager() as manager:
        q = manager.Queue()
        
        is_dones = [False for i in range(nprocess)]
        is_dones = manager.list(is_dones)
        processes = [mp.Process(target=run_process, args=(i, q, is_dones)) for i in range(nprocess)]
        
        [p.start() for p in processes]

        try:
            running_context(q)
            q.join()
            
            for i in range(nprocess):
                is_dones[i] = True
        except Exception:
            raise
        finally:
            [p.join() for p in processes]


def gpu_parallel(process_func, *args,**kwargs):
    """
    Parallel processes on all gpus

    Args:
        process_func (function): function with the first argument is device id
    
    Return:
        list of result on each gpu
    """   
    def proxy_func(device_id: int, queue: mp.Queue):
        result = process_func(device_id)

        if result is not None:
            queue.put_nowait((device_id, result))

    with torch_mp.Manager() as manager:
        devices = torch.cuda.device_count()

        q = manager.Queue()
        processes = [torch_mp.Process(target=proxy_func, args=(d, q, *args), kwargs=kwargs) for d in range(devices)]

        [p.start() for p in processes]
        [p.join() for p in processes]

        return list(iter(q.get, None))
