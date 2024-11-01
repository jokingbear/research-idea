import warnings
import networkx as nx
import inspect

from ..functional import auto_map_func


class ObjectFactory(dict):

    def __init__(self, append=False):
        super().__init__()

        self.append = append

    def register(self, *names, verbose=True):
        return object_map(self, names, self.append, verbose)
    
    def shared_init(self, *args, **kwargs):
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
    
    def normal_init(self, *args, **kwargs):
        return {k: initiator(*args, **kwargs) for k, initiator in self.items()}

    def mapped_init(self, mapped_args:dict):
        results = {}
        for k, initiator in self.items():
            if k in mapped_args:
                if isinstance(initiator, list):
                    results[k] = [auto_map_func(init)(arg) for init, arg in zip(initiator, mapped_args[k])]
                else:
                    results[k] = auto_map_func(initiator)(mapped_args[k])

        return results
    
    def inject_dependencies(self, leaves_args:dict):
        assert not self.append, 'inject_dependencies does not support list init at the moment'
        dependency_graph = nx.DiGraph()

        for key, initiator in self.items():
            dependency_graph.add_node(key, init=initiator)
        
        leaf_factories = ObjectFactory()
        initiateds = {}
        for n, attrs in dependency_graph.nodes.items():
            init = attrs['init']
            if n in leaves_args:
                leaf_factories[n] = init

            arg_specs = inspect.getfullargspec(init)
            args = [a for a in arg_specs.args if a != 'self']
            
            count = 0
            for arg in args:
                if arg in dependency_graph:
                    dependency_graph.add_edge(n, arg)
                elif arg in leaves_args:
                    initiateds[arg] = leaves_args[arg]
                elif count > 0:
                    raise ValueError(f'dependency "{arg}" has not been registered for {init}')
                
                count += 1
        
        initiateds.update(leaf_factories.mapped_init(leaves_args))
        for n, attrs in dependency_graph.nodes.items():
            init = attrs['init']
            _recursive_init(dependency_graph, n, init, initiateds)
        
        return {k: v for k, v in initiateds.items() if k in dependency_graph}


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


def _recursive_init(graph:nx.DiGraph, key, init, results):
    if key not in results:
        arg_specs = inspect.getfullargspec(init)
        arg_names = [a for a in arg_specs.args if a != 'self']
        
        args = {}
        for a in arg_names:
            if a in graph:
                _recursive_init(graph, a, graph[a].get('init'), results)
            
            args[a] = results[a]

        results[key] = init(**args)
