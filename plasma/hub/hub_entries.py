import inspect as insp

from importlib import import_module
from .utils import get_base


class HubEntries:

    def __init__(self, path, default_file):
        base = get_base(path)
        self.module = import_module(f"{base}.{default_file}")

    def load(self, entry_name, *args, **kwargs):
        function = getattr(self.module, entry_name)

        assert insp.isfunction(function), f"{function} is not a function"

        return function(*args, **kwargs)

    def list_entries(self):
        function_names = [name for name, _ in insp.getmembers(self.module, insp.isfunction)]

        return function_names

    def list_specs(self, entry_name):
        function = getattr(self.module, entry_name)

        assert insp.isfunction(function), f"{function} is not a method"

        spec = insp.getfullargspec(function)
        return spec
