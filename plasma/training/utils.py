import torch
import torch.onnx as onnx
import onnxruntime as runtime

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
        return [x.type(dtype).to(device) for x in xs]
    else:
        x = xs.type(dtype).to(device)
        return x


def get_inputs_labels(xy, x_type, x_device, y_type, y_device):
    if type(xy) in {tuple, list}:
        x = to_device(xy[0], dtype=x_type, device=x_device)
        y = to_device(xy[1], dtype=y_type, device=y_device)

        return x, y
    else:
        x = to_device(xy, dtype=x_type, device=x_device)

        return x, x


def eval_model(model):
    model.eval()

    return torch.no_grad()


def iterate_numpies(*arr, batch_size=32):
    n = arr[0].shape[0]

    n_iter = n // batch_size
    if n % batch_size != 0:
        n_iter += 1
    
    for p in range(n_iter):
        result = [a[p * batch_size:(p + 1) * batch_size] for a in arr]
        yield result[0] if len(result) == 1 else result


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


def load_onnx(path, output_names=None):
    session = runtime.InferenceSession(path)

    def predict(*args):
        args = {f"input_{i}": a for i, a in enumerate(args)}
        preds = session.run(output_names, args)

        return preds

    return predict
