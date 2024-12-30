import inspect
from ..functional import AutoPipe
from .class_registrator import ObjectFactory


class DependencyInjector(AutoPipe):

    def __init__(self, factory:ObjectFactory, strict=False):
        super().__init__()

        assert not factory.append, 'DependencyInjector does not support append type registration at the moment'
        self.factory = factory
        self.strict = strict

    def run(self, init_args:dict, *names) -> dict:
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
            
            args = {}
            for arg in arg_names:
                if arg in init_args:
                    args[arg] = init_args[arg]
                elif arg in self.factory:
                    self._recursive_init(arg, self.factory[arg], object_dict, init_args)
                
                    if arg not in object_dict:
                        if not self.strict:
                            print(f'{key} does not have dependency {arg}, skipping key {key}')
                            break

                        raise KeyError(f'{arg} is not registered in object_dict or factory')
                    args[arg] = object_dict[arg]
                else:
                    raise KeyError(f'{arg} is not in init_args or factory')
            
            if len(args) == len(arg_names):
                object_dict[key] = object_initiator(**args)
