import torch

default_device = "cpu"
default_type = torch.float


def to_device(xs, dtype=None, device=None):
    device = device or default_device
    dtype = dtype or default_type

    if type(xs) in {list, tuple}:
        return [to_device(x, dtype, device) for x in xs]
    elif isinstance(xs, dict):
        return {k: to_device(xs[k], dtype, device) for k in xs}
    else:
        x = xs.to(device).type(dtype)
        return x


def get_batch_tensors(batch_values, x_type, x_device, y_type, y_device):
    if type(batch_values) in {tuple, list}:
        x = to_device(batch_values[0], dtype=x_type, device=x_device)
        y = to_device(batch_values[1], dtype=y_type, device=y_device)

        return x, y
    else:
        x = to_device(batch_values, dtype=x_type, device=x_device)

        return x, x


def get_dict(values, prefix=None, name=None):
    prefix = prefix or ""
    name = name or "loss"

    if isinstance(values, dict):
        d = {prefix + k: float(values[k]) for k in values}
    else:
        d = {prefix + name: float(values)}

    return d
