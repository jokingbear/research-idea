import inspect as insp
import sys

from importlib import import_module
from pathlib import Path


class HubEntries:

    def __init__(self, absolute_path, module_name):
        base = Path(absolute_path)
        sys.path.append(str(base))
        self.module = import_module(module_name.replace(".py", ""))

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
        """
        list all available entries
        :return: list of entries
        """
        function_names = [name for name, _ in insp.getmembers(self.module, insp.isfunction)]

        return function_names

    def inspect(self, entry_name):
        """
        inspect args and kwargs of an entry
        :param entry_name: name of the entry
        :return: argspec object
        """
        function = getattr(self.module, entry_name)

        assert insp.isfunction(function), f"{function} is not a function"

        spec = insp.getfullargspec(function)
        return spec
