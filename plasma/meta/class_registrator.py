import warnings


class ObjectFactory(dict):

    def register(self, name=None, verbose=True):
        return object_map(self, name, verbose)


class object_map:

    def __init__(self, factory: ObjectFactory, name, verbose):
        self._factory = factory
        self.name = name
        self.verbose = verbose

        if self.name in self._factory:
            warnings.warn(f'many entry points for: {self.name}, overriding the current one')

    def __call__(self, func_or_class):
        name = self.name or func_or_class.__qualname__
        self._factory[name] = func_or_class

        if self.verbose:
            print(f'registered {name}: {func_or_class}')
        return func_or_class
