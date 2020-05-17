import torch
import torch.onnx as onnx
import torch.utils.data as data

import numpy as np

from tqdm import tqdm, tqdm_notebook as tqdm_nb

on_notebook = True
default_device = "cpu"
default_type = torch.float


def get_tqdm():
    return tqdm_nb if on_notebook else tqdm


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


def eval_modules(*modules):
    [m.eval() for m in modules]

    return torch.no_grad()


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
                opset_version=10,)


def get_batch_iterator(*arrs, batch_size=32):
    arrs = [np.array(arr) for arr in arrs]
    n = min([arr.shape[0] for arr in arrs])

    iterations = n // batch_size
    mod = n % batch_size

    if mod != 0:
        iterations += 1

    if len(arrs) == 1:
        return [arrs[0][i * batch_size:(i + 1) * batch_size] for i in range(iterations)]
    else:
        return [[arr[i * batch_size:(i + 1) * batch_size] for arr in arrs] for i in range(iterations)]


def get_loader(*arrs, mapper=None, batch_size=32, pin_memory=True, workers=None):
    n = min([len(a) for a in arrs])
    workers = workers or batch_size // 2
    mapper = mapper or (lambda *args: args[0] if len(args) == 1 else args)

    class Data(data.Dataset):

        def __len__(self):
            return n

        def __getitem__(self, idx):
            items = [a[idx] for a in arrs]

            return mapper(*items)

    return data.DataLoader(Data(), batch_size, pin_memory=pin_memory, num_workers=workers)
