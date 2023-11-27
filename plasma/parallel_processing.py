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
                     auto_func=True, **kwargs):
    """
    Parallel iter an array

    Args:
        arr: array to be iterated
        iter_func (function arg): function to be called for each data, signature (idx, arg) or arg
        context_manager: context manager from multiprocessing module
        workers (int, optional): number of worker to run. Defaults to 8.
        batchsize: batch size to run in parallel.
        use_index (bool, optional): whether to add index to each call of iter func. Defaults to False.
        auto_func: whether to treat tuple or list as args
    Returns:
        list: list of result for each element in the array
    """    

    if auto_func:
        iter_func = auto_map_func(iter_func)
    if len(kwargs) > 0:
        iter_func = partials(iter_func, **kwargs)

    iterator = _build_iterator(arr, use_index)

    context_manager = context_manager or mp
    with context_manager.Pool(workers) as pool:
        jobs = pool.imap(iter_func, iterator, batchsize)
        results = [j for j in get_tqdm(jobs, total=len(arr))]

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


def _queue_item_process(in_queue: mp.Queue, func, progress, total_results):
    while True:
        i, x = in_queue.get()
        results = func(x)

        with progress.get_lock():
            progress.value += 1

        total_results.append((i, results))


def _check_progress(progress, desc='running'):
    current_value = progress.value

    with get_tqdm(desc=desc) as pbar:
        while True:
            new_value = progress.value
            if new_value != current_value:
                pbar.update(new_value - current_value)
                current_value = new_value


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

    with (context_manager.Queue() if context_manager is not None else mp.Queue()) as q:
        processes = [mp.Process(target=run_process, args=(q,)) for _ in range(nprocess)]
        
        [p.start() for p in processes]

        try:
            running_func(context_manager, q)
            q.join()
        except Exception:
            raise
        finally:
            [p.join() for p in processes]
