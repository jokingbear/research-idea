import torch

default_device = "cpu"
default_type = torch.float


def to_device(xs, dtype=None, device=None):
    device = device or default_device
    dtype = dtype or default_type

    if type(xs) in {list, tuple}:
        return [x.to(device).type(dtype) for x in xs]
    else:
        x = xs.to(device).type(dtype)
        return x


def get_inputs_labels(xy, x_type, x_device, y_type, y_device):
    if type(xy) in {tuple, list}:
        x = to_device(xy[0], dtype=x_type, device=x_device)
        y = to_device(xy[1], dtype=y_type, device=y_device)

        return x, y
    else:
        x = to_device(xy, dtype=x_type, device=x_device)

        return x, x


def get_dict(values, prefix=None, name=None):
    prefix = prefix or ""
    name = name or "loss"

    d = {prefix + k: float(values[k]) for k in values} if isinstance(values, dict) \
        else {prefix + (name or "loss"): float(values)}

    return d
