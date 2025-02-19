import inspect
from ..functional import AutoPipe
from .class_registrator import ObjectFactory


class DependencyInjector(AutoPipe):

    def __init__(self, factory:ObjectFactory, strict=False):
        super().__init__()

        assert not factory.append, 'DependencyInjector does not support append type registration at the moment'
        self.factory = factory
        self.strict = strict

    def run(self, *names, **init_args) -> dict:
        if len(names) == 0:
            names = {*self.factory}
        else:
            names = {*names}

        object_dict = {}
        for key, object_initiator in self.factory.items():
            if key in names: 
                self._recursive_init(key, object_initiator, object_dict, init_args)
                
        return {k:v for k, v in object_dict.items() if k in names}

    def _recursive_init(self, key, object_initiator, object_dict:dict, init_args:dict):
        if key not in object_dict:
            argspecs = inspect.getfullargspec(object_initiator)
            arg_names = [a for a in argspecs.args if a != 'self']
            defaults = argspecs.defaults or []
            arg_defaults = {name:value for name, value in zip(argspecs.args[::-1], defaults[::-1])}

            arg_maps = {}
            for arg in arg_names:
                arg_object = _NotInitialized

                if arg in init_args:
                    arg_object = init_args[arg]
                elif arg in self.factory:
                    self._recursive_init(arg, self.factory[arg], object_dict, init_args)
                    arg_object = object_dict.get(arg, _NotInitialized)
                elif arg in arg_defaults:
                    arg_object = arg_defaults[arg]

                if arg_object is _NotInitialized:
                    error_message = f'{arg} is not in init_args or dependency graph at key: {key}'
                    if not self.strict:
                        print(error_message)
                        break
                    else:
                        raise KeyError(error_message)

                arg_maps[arg] = arg_object

            if len(arg_maps) == len(arg_names):
                object_dict[key] = object_initiator(**arg_maps)


class _NotInitialized:
    pass
