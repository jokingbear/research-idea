import pandas as pd
import multiprocessing as mp

from .functional import auto_map_func
from plasma import get_tqdm
from queue import Empty


def create_context():
    return mp.Manager()


def parallel_iterate(arr, iter_func, context_manager: mp.Manager, workers=8, use_index=False, **kwargs):
    """
    Parallel iter an array

    Args:
        arr: array to be iterated
        iter_func (function arg): function to be called for each data, signature (idx, arg) or arg
        context_manager: context manager from multiprocessing module
        workers (int, optional): number of worker to run. Defaults to 8.
        use_index (bool, optional): whether to add index to each call of iter func. Defaults to False.
    Returns:
        list: list of result for each element in the array
    """    

    with (context_manager.Pool(workers) if context_manager is not None else mp.Pool(workers)) as p:
        if isinstance(arr, zip):
            jobs = [p.apply_async(iter_func, args=(i, *arg) if use_index else arg, kwds=kwargs) for i, arg in enumerate(arr)]
        elif isinstance(arr, pd.DataFrame):
            jobs = [p.apply_async(iter_func, args=(i, row) if use_index else (row,), kwds=kwargs) for i, row in arr.iterrows()]
        else:
            jobs = [p.apply_async(iter_func, args=(i, arg) if use_index else (arg,), kwds=kwargs) for i, arg in enumerate(arr)]
        results = [j.get() for j in get_tqdm(jobs)]
    return results


def process_queue(running_func, post_process_func, context_manager: mp.Manager,
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

    with context_manager.Queue() as q:
        processes = [mp.Process(target=run_process, args=(q,)) for _ in range(nprocess)]
        
        [p.start() for p in processes]

        try:
            running_func(context_manager, q)
            q.join()
        except Exception:
            raise
        finally:
            [p.join() for p in processes]
