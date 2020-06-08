import inspect as insp

from importlib import import_module
from .utils import get_base


default_file = "hubconfig"


def load(path, method_name, *args, **kwargs):
    base = get_base(path)

    module = import_module(f"{base}.{default_file}")
    method = getattr(module, method_name)

    assert insp.ismethod(method), f"{method} is not a method"

    return method(*args, **kwargs)


def list_entries(path):
    base = get_base(path)

    module = import_module(f"{base}.{default_file}")
    method_names = [name for name, _ in insp.getmembers(module, insp.ismethod)]

    return method_names


def list_specs(path, method_name):
    base = get_base(path)

    module = import_module(f"{base}.{default_file}")
    method = getattr(module, method_name)

    assert insp.ismethod(method), f"{method} is not a method"

    spec = insp.getfullargspec(method)
    print(spec)
