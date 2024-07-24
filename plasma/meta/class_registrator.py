import warnings


class ObjectFactory(dict):

    def register(self, *names, verbose=True):
        return object_map(self, names, verbose)


class object_map:

    def __init__(self, factory: ObjectFactory, names, verbose):
        self._factory = factory
        self.names = names
        self.verbose = verbose

        if self.names in self._factory:
            warnings.warn(f'many entry points for: {self.names}, overriding the current one')

    def __call__(self, func_or_class):
        names = self.names
        if len(names) == 0:
            names = [func_or_class.__qualname__]

        for name in names:
            self._factory[name] = func_or_class

            if self.verbose:
                print(f'registered {name}: {func_or_class}')

        return func_or_class
