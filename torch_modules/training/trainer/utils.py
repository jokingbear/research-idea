import torch

from tqdm import tqdm, tqdm_notebook as tqdm_nb

on_notebook = True
default_device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")


def get_pbar():
    return tqdm_nb if on_notebook else tqdm


def to_device(xs, dtype, device):
    device = device or default_device

    if device == "cpu":
        return xs

    if type(xs) is list:
        return [x.type(dtype).to(device) for x in xs]
    else:
        return xs.type(dtype).to(device)
