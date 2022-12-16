import multiprocessing as mp
import os

import torch
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from .data.adhoc_data import AdhocData


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
        else:
            jobs = [p.apply_async(iter_func, args=(i, arg) if use_index else (arg,), kwds=kwargs) for i, arg in enumerate(arr)]
        results = [j.get() for j in get_progress(jobs)]
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


def torch_parallel_iterate(arr, iteration_func, loader_func=None, cleanup_func=None, batch_size=32, workers=2, loader_kwargs={}, 
                            dtype=torch.float16, **kwargs):
    """
    parallel iterate over arr using torch
    :param arr: array to be iterate over
    :param iteration_func: what to compute on gpu
    :param loader_func: how to load the data, the default is identity
    :param cleanup_func: how to clean up the data after each iteration
    :batch_size: batch size to load on gpus
    :workers: number of worker for torch loader
    :loader_kwargs: additional param for torch loader
    :dtype: cast type on gpu
    :kwargs: additional arg for iteration_func
    """

    if loader_func is None:
        loader_func = lambda x: x

    loader = get_loader(arr, loader_func, batch_size=batch_size, workers=workers, **loader_kwargs)

    class TempModule(torch.nn.Module):

        def __init__(self):
            super().__init__()

            for k in kwargs:
                value = kwargs[k]
                
                if issubclass(type(value), torch.nn.Module):
                    setattr(self, k, value)
                elif not isinstance(k, torch.Tensor):
                    value = torch.tensor(value, dtype=dtype)
                
                self.register_parameter(k, torch.nn.Parameter(value, requires_grad=False))
        
        def forward(self, x):
            return iteration_func(x, **{k: getattr(self, k) for k in kwargs})
    
    m = TempModule()
    m = torch.nn.DataParallel(m)
    m = m.cuda()
    results = []

    with torch.no_grad():
        for i, d in get_progress(enumerate(loader), total=len(loader)):
            d = d.type(dtype)
            result = m(d.cuda())
            
            if cleanup_func is not None:
                cleanup_func(i, result)
            else:
                results.append(result)
    
    return results


def process_queue(running_context, process_func, nprocess=50, infinite_loop=True):
    """
    create a queue with nprocess to resolve that queue
    :param running_context: running function, receive queue as input
    :param process_func: function to process queue item
    :param nprocess: number of independent process
    :param infinite_loop: number of worker to run
    :return queue, n processes
    """
    def run_process(i, queue: mp.Queue):
        print(f'process {i} started')
        condition = True

        while condition:
            item = queue.get()

            if isinstance(item, tuple):
                process_func(*item)
            elif isinstance(item, dict):
                process_func(**item)
            else:
                process_func(item)

            condition &= infinite_loop

    q = mp.Manager().Queue()
    processes = [mp.Process(target=run_process, args=(i, q)) for i in range(nprocess)]

    [p.start() for p in processes]

    try:
        running_context(q)
    except:
        raise
    finally:
        [p.terminate() for p in processes]
