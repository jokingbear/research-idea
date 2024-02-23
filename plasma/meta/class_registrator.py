import warnings

from sympy import N


class ObjectFactory(dict):

    def register(self, name=None):
        return object_map(self, name)


class object_map:

    def __init__(self, factory: ObjectFactory, name):
        self._factory = factory
        self.name = name

        if self.name in self._factory:
            warnings.warn(f'many entry points for: {self.name}, overriding the current one')

    def __call__(self, func_or_class):
        name = self.name or func_or_class.__qualname__
        self._factory[name] = func_or_class
        return func_or_class
