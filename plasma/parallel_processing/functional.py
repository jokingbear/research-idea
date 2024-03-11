import pandas as pd
import multiprocessing as mp

from plasma.functional import partials
from tqdm.auto import tqdm
from warnings import warn


def parallel_iterate(arr, iter_func, context_manager: mp.Manager = None, workers=8, batchsize=100, use_index=False,
                     estimate_length=None, progress=True, **kwargs):
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
        progress: show progress bar
    Returns:
        list: list of result for each element in the array
    """
    if len(kwargs) > 0:
        iter_func = partials(iter_func, **kwargs)

    warn('parallel_iterate is deprecated, will be remove in the next release, use tqdm.contrib.concurrent.process_map instead')
    iterator = _build_iterator(arr, use_index)

    context_manager = context_manager or mp
    with context_manager.Pool(workers) as pool:
        jobs = pool.imap(iter_func, iterator, batchsize)
        results = [j for j in tqdm(jobs, total=estimate_length or len(arr), show=progress)]

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
