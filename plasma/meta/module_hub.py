import inspect as insp
import sys

from importlib import import_module
from .entry_factory import get_module_entry


class ModuleHub:

    def __init__(self, absolute_path, module_name):
        absolute_path = str(absolute_path)
        if absolute_path not in sys.path:
            sys.path.append(absolute_path)

        self.module = import_module(module_name)

    def load(self, entry_name, *args, **kwargs):
        """
        load a function from entry file

        :param entry_name: function name
        :param args: args to input into the function
        :param kwargs: kwargs to input into the function
        :return:
        """
        function = self[entry_name]

        assert insp.isfunction(function) or insp.isclass(function), \
            f"{function} is not a function or a class"

        return function(*args, **kwargs)

    def list(self):
        """
        list all available entries
        :return: list of entries
        """
        function_names = [name for name, _ in insp.getmembers(self.module, insp.isfunction)]
        class_names = [name for name, _ in insp.getmembers(self.module, insp.isclass)]

        return function_names + class_names

    def inspect(self, entry_name):
        """
        inspect args and kwargs of an entry
        :param entry_name: name of the entry
        :return: argspec object
        """
        function = self[entry_name]

        assert insp.isfunction(function) or insp.isclass(function), \
            f"{function} is not a function or a class"

        spec = insp.getfullargspec(function)
        return spec

    @property
    def name(self):
        return self.module.__name__

    @property
    def entry(self):
        return get_module_entry(self.name)

    def load_entry(self, *args, **kwargs):
        entry = self.entry
        assert entry is not None, 'there\'s no marked entry'

        return self.entry(*args, **kwargs)

    def __getitem__(self, name):
        return getattr(self.module, name)

    def __repr__(self):
        return f'module={self.module}'

    def __getattr__(self, name):
        return getattr(self.module, name)
