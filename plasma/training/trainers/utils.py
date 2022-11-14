import torch

default_device = "cpu"
default_type = torch.float


def to_device(xs, dtype=None, device=None):
    device = device or default_device
    dtype = dtype or default_type
    dtype = torch.float if dtype == "float" else torch.long if dtype == "long" else dtype

    if type(xs) in {list, tuple}:
        return [to_device(x, dtype, device) for x in xs]
    elif isinstance(xs, dict):
        return {k: to_device(xs[k], dtype, device) for k in xs}
    else:
        x = xs.to(device).type(dtype)
        return x


def get_batch_tensors(batch_values, dtype, device):
    if isinstance(batch_values, tuple):
        return tuple([to_device(v, dtype, device) for v in batch_values])
    elif isinstance(batch_values, list):
        return [to_device(v, dtype, device) for v in batch_values]
    elif isinstance(batch_values, dict):
        return {k: to_device(batch_values[k], dtype, device) for k in batch_values}

    raise 'only support type tuple, list and dict'


def get_dict(values, prefix=None, name=None):
    prefix = prefix or ""
    name = name or "loss"

    if isinstance(values, dict):
        d = {prefix + k: float(values[k]) for k in values}
    else:
        d = {prefix + name: float(values)}

    return d
