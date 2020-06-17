import inspect as insp

from importlib import import_module
from .utils import get_base


class HubEntries:

    def __init__(self, path, default_file):
        base = get_base(path)
        self.module = import_module(f"{base}.{default_file}")

    def load(self, entry_name, *args, **kwargs):
        """
        load a function from entry file

        :param entry_name: function name
        :param args: args to input into the function
        :param kwargs: kwargs to input into the function
        :return:
        """
        function = getattr(self.module, entry_name)

        assert insp.isfunction(function), f"{function} is not a function"

        return function(*args, **kwargs)

    def list(self):
        function_names = [name for name, _ in insp.getmembers(self.module, insp.isfunction)]

        return function_names

    def inspect(self, entry_name):
        function = getattr(self.module, entry_name)

        assert insp.isfunction(function), f"{function} is not a function"

        spec = insp.getfullargspec(function)
        return spec
