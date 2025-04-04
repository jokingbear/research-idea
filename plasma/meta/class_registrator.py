import warnings
import networkx as nx
import inspect

from ..functional import auto_map_func


class ObjectFactory[K,V](dict[K, type[V]]):

    def __init__(self, append=False):
        super().__init__()

        self.append = append

    def register(self, *names, verbose=True):
        return object_map(self, names, self.append, verbose)
    
    def shared_init(self, *args, **kwargs) -> dict[K, V]:
        obj_dict = {}
        results = {}
        for k, initiator in self.items():
            if isinstance(initiator, list):
                for init in initiator:
                    if init not in obj_dict:
                        obj_dict[init] = init(*args, **kwargs)
                results[k] = [obj_dict[i] for i in initiator]
            else:
                if initiator not in obj_dict:
                    obj_dict[initiator] = initiator(*args, **kwargs)
                results[k] = obj_dict[initiator]

        return results
    
    def normal_init(self, *args, **kwargs) -> dict[K, V]:
        return {k: initiator(*args, **kwargs) for k, initiator in self.items()}

    def mapped_init(self, mapped_args:dict) -> dict[K, V]:
        results = {}
        for k, initiator in self.items():
            if k in mapped_args:
                if isinstance(initiator, list):
                    results[k] = [auto_map_func(init)(arg) for init, arg in zip(initiator, mapped_args[k])]
                else:
                    results[k] = auto_map_func(initiator)(mapped_args[k])

        return results


class object_map:

    def __init__(self, factory: ObjectFactory, names, append, verbose):
        self._factory = factory
        self.names = names
        self.append = append
        self.verbose = verbose

        if self.names in self._factory:
            warnings.warn(f'many entry points for: {self.names}, overriding the current one')

    def __call__(self, func_or_class):
        names = self.names
        if len(names) == 0:
            names = [func_or_class.__qualname__]

        for name in names:
            target = func_or_class
            if self.append:
                target = self._factory.get(name, [])
                target.append(func_or_class)

            self._factory[name] = target

            if self.verbose:
                print(f'registered {name}: {func_or_class}')

        return func_or_class
