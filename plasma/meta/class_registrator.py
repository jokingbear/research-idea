import warnings
import pandas as pd


class ObjectFactory(dict):

    def register(self, *names, verbose=True):
        return object_map(self, names, verbose)
    
    def shared_init(self, *args, **kwargs):
        obj_dict = {}
        results = {}
        for k, initiator in self.items():
            if initiator not in obj_dict:
                obj_dict[initiator] = initiator(*args, **kwargs)
            
            results[k] = obj_dict[initiator]

        return results
    
    def normal_init(self, *args, **kwargs):
        return {k: initiator(*args, **kwargs) for k, initiator in self.items()}


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
