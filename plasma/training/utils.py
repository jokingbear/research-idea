import multiprocessing as mp
import os

import numpy as np
import torch
import torch.onnx as onnx
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


# TODO: refactor for dict outputs
def save_onnx(path, model, *input_shapes, device="cpu"):
    model = model.eval()
    args = [torch.ones([1, *shape], requires_grad=True, device=device) for shape in input_shapes]
    outputs = model(*args)

    if torch.is_tensor(outputs):
        outputs = [outputs]

    input_names = [f"input_{i}" for i, _ in enumerate(args)]
    output_names = [f"output_{i}" for i, _ in enumerate(outputs)]

    onnx.export(model, tuple(args), path,
                export_params=True, verbose=True, do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={n: {0: "batch_size"} for n in input_names + output_names},
                opset_version=10, )


def parallel_iterate(arr, iter_func, batch_size=32, workers=8):
    """
    parallel iterate array
    :param arr: array to be iterated
    :param iter_func: function to be called for each data
    :param batch_size: batch size to run in parallel
    :param workers: number of worker to run
    """
    arr = np.array(arr)
    n = arr.shape[0]

    iterations = n // batch_size
    mod = n % batch_size

    if mod != 0:
        iterations += 1

    elems = [arr[i * batch_size:(i + 1) * batch_size] for i in range(iterations)]

    pool = mp.Pool(workers)
    for e in get_tqdm(iterable=elems):
        pool.map(iter_func, e)


def get_loader(*arrs, mapper=None, batch_size=32, pin_memory=True, workers=20):
    n = min([len(a) for a in arrs])
    workers = workers or batch_size // 2

    class Data(data.Dataset):

        def __len__(self):
            return n

        def __getitem__(self, idx):
            items = [a[idx] for a in arrs]

            if mapper is not None:
                return mapper(*items)
            elif len(items) == 1:
                return items[0]

            return items

    return data.DataLoader(Data(), batch_size, shuffle=False, drop_last=False,
                           pin_memory=pin_memory, num_workers=workers)


def visible_devices(*device_ids):
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids)
