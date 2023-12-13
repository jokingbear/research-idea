import ctypes
import time

import pandas as pd
import multiprocessing as mp

from .functional import auto_map_func, partials
from plasma import get_tqdm
from queue import Empty


def create_context():
    return mp.Manager()


def parallel_iterate(arr, iter_func, context_manager: mp.Manager = None, workers=8, batchsize=100, use_index=False,
                     estimate_length=None, **kwargs):
    """
    Parallel iter an array

    Args:
        arr: array to be iterated
        iter_func (function arg): function to be called for each data, signature (idx, arg) or arg
        context_manager: context manager from multiprocessing module
        workers (int, optional): number of worker to run. Defaults to 8.
        batchsize: batch size to run in parallel.
        use_index (bool, optional): whether to add index to each call of iter func. Defaults to False.
        estimate_length: estimate length of the iteration
    Returns:
        list: list of result for each element in the array
    """    
    if len(kwargs) > 0:
        iter_func = partials(iter_func, **kwargs)

    iterator = _build_iterator(arr, use_index)

    context_manager = context_manager or mp
    with context_manager.Pool(workers) as pool:
        jobs = pool.imap(iter_func, iterator, batchsize)
        results = [j for j in get_tqdm(jobs, total=estimate_length or len(arr))]

    return results


def _build_iterator(array, use_index):
    if isinstance(array, pd.DataFrame):
        for idx, row in array.iterrows():
            if use_index:
                yield idx, row
            else:
                yield row
    else:
        for idx, args in enumerate(array):
            if use_index:
                if isinstance(args, tuple):
                    yield idx, *args
                else:
                    yield idx, args
            else:
                yield args


def process_queue(running_func, post_process_func, context_manager: mp.Manager = None,
                  nprocess=10, infinite_loop=True, timeout=20):
    """
    Create a queue and process item asynchronously

    Args:
        running_func (function (queue, manager) -> None): a function that put item inside a queue.
        post_process_func (function (queue_item) -> None): function to precess queue. 
        This function will be run asynchronously on nprocess.
        nprocess (int, optional): number of parallel processes. Defaults to 50.
        context_manager: manager of multiprocessing context
        infinite_loop (bool, optional): whether to run process_func in infinite loop. Defaults to True.
        timeout (int, optional): a period (second) a process should wait for a queue. Defaults to 20s.
    """    

    post_process_func = auto_map_func(post_process_func)

    timeout = timeout or 20

    def run_process(queue: mp.Queue):
        condition = True

        while condition:
            try:
                item = queue.get(timeout=timeout)

                post_process_func(item)

                queue.task_done()
                condition &= infinite_loop
            except Empty:
                return

    creator = context_manager or mp
    with creator.Queue() as q:
        processes = [creator.Process(target=run_process, args=(q,)) for _ in range(nprocess)]
        
        [p.start() for p in processes]

        try:
            running_func(context_manager, q)
            q.join()
        except Exception:
            raise
        finally:
            [p.terminate() for p in processes]
