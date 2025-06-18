import inspect
import networkx as nx
import re
import pandas as pd

from ..functional import State


class DependencyInjector(State):

    def __init__(self, stateful=False):
        super().__init__()

        self._dep_graph = nx.DiGraph()
        self.stateful = stateful

    def run(self, *names, **init_args) -> dict:
        if len(names) == 0:
            names = self._dep_graph.nodes
        
        names = list(names)
        object_dict = {}
        for n in names:
            self._recursive_init(n, object_dict, init_args)
                
        return pd.Series({n: object_dict.get(n, _NotInitialized) for n in names}).loc[names]
    
    def add_dependency(self, name, value, as_singleton=False):
        assert name[0] != '_', 'dependency cannot start with _'
        assert as_singleton or callable(value), 'depdency should be callable'
        
        if name in self._dep_graph:
            neighbors = [*self._dep_graph.neighbors(name)]
            self._dep_graph.remove_edges_from([(name, n) for n in neighbors])
        
        if as_singleton:
            self._dep_graph.add_node(name, value=value)    
        else:
            self._dep_graph.add_node(name, initiator=value)
            
            parameters = inspect.signature(value).parameters
            for arg_name, p in parameters.items():
                if arg_name != 'self':
                    self._dep_graph.add_node(arg_name)
                    self._dep_graph.add_edge(name, arg_name)
                    
                    if p.default is not inspect.Parameter.empty:
                        self._dep_graph.add_node(arg_name, value=p.default)

        return self

    def decorate_dependency(self, name):

        def decorator(func_or_class):
            self.add_dependency(name, func_or_class)
            return func_or_class

        return decorator

    def merge(self, injector):
        assert isinstance(injector, DependencyInjector), 'injector must be an DependencyInjector instance'
        self._dep_graph = nx.compose(self._dep_graph, injector._dep_graph)

        return self

    def duplicate(self, current_name:str, new_name:str):
        assert current_name in self._dep_graph, 'current name must be in dep graph'
        assert new_name not in self._dep_graph, 'new name must not be in dep graph'
        
        node = self._dep_graph.nodes[current_name]
        neighbors = [*self._dep_graph.successors(current_name)]
        self._dep_graph.add_node(new_name, **node)
        for n in neighbors:
            self._dep_graph.add_edge(new_name, n)
        
        return self
    
    def _recursive_init(self, key, object_dict:dict, init_args:dict):
        if key not in object_dict and key in self._dep_graph:
            if 'value' in self._dep_graph.nodes[key]:
                object_dict[key] = self._dep_graph.nodes[key]['value']
            else:
                arg_maps = {}
                for arg in self._dep_graph.neighbors(key):
                    if arg in init_args:
                        arg_object = init_args[arg]
                    else:
                        node_attributes = self._dep_graph.nodes[arg]
                        if 'value' in node_attributes:
                            arg_object = node_attributes['value']
                        else:
                            self._recursive_init(arg, object_dict, init_args)
                            arg_object = object_dict.get(arg, _NotInitialized)

                    if arg_object is _NotInitialized:
                        error_message = f'{arg} is not in init_args or dependency graph at key: {key}'
                        raise KeyError(error_message)

                    arg_maps[arg] = arg_object

                if len(arg_maps) == self._dep_graph.out_degree(key):
                    try:
                        object_dict[key] = self._dep_graph.nodes[key]['initiator'](**arg_maps)
                        if self.stateful:
                            self._dep_graph.add_node(key, value=object_dict[key])
                    except Exception as e:
                        raise RuntimeError(f'error at {key}') from e

    def __repr__(self):
        lines = []
        prefix = '|'
        indent = '--'
        for r in self._dep_graph:
            if self._dep_graph.in_degree(r) == 0:
                lines.append(_render_node(self._dep_graph, r, prefix, indent))
                lines.append('-' * 100)
        text = '\n'.join(lines)
        pattern = f'{prefix}{indent}'.replace('|', r'\|')
        text = re.sub(rf'({pattern}){{1,}}', lambda m: m.group(0).replace(prefix + indent, ' ' * (len(indent) + 1)) + prefix + indent, text)
        return text

    def release(self):
        for n, attr in self._dep_graph.nodes.items():
            if 'value' in attr and self._dep_graph.out_degree(n) > 0:
                del attr['value']


class _NotInitialized:
    pass


def _render_node(graph:nx.DiGraph, key, prefix='|', indent=' ' * 2):
    node = graph.nodes[key]
    if 'value' in node:
        lines = [f'{key}={type(node["value"])}']
    else:
        lines = [key]
        for n in graph.neighbors(key):
            rendered_lines = _render_node(graph, n, prefix, indent)
            rendered_lines = rendered_lines.split('\n')
            rendered_lines = [prefix + indent + t for t in rendered_lines]
            lines.extend(rendered_lines)
    
    return '\n'.join(lines)
