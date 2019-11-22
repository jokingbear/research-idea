import torch

from tqdm import tqdm, tqdm_notebook as tqdm_nb

on_notebook = True
default_device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
default_type = torch.float


def get_tqdm():
    return tqdm_nb if on_notebook else tqdm


def to_device(xs, dtype=None, device=None, return_array=True):
    device = device or default_device
    dtype = dtype or default_type

    if device == "cpu":
        return xs

    if type(xs) in {list, tuple}:
        return [x.type(dtype).to(device) for x in xs]
    else:
        x = xs.type(dtype).to(device)
        return [x] if return_array else x
