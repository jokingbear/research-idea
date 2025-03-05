import inspect
import networkx as nx
import re

from ..functional import AutoPipe


class DependencyInjector(AutoPipe):

    def __init__(self, strict=False):
        super().__init__()

        self.strict = strict
        self._dep_graph = nx.DiGraph()

    def run(self, *names, **init_args) -> dict:
        if len(names) == 0:
            names = {*self._dep_graph.nodes}
        else:
            names = {*names}

        object_dict = {}
        for n in names:
            self._recursive_init(n, object_dict, init_args)
                
        return {n: object_dict.get(n, _NotInitialized) for n in names}
    
    def add_dependency(self, name, value):
        assert name[0] != '_', 'dependency cannot start with _'
        assert callable(value), 'depdency should be callable'
        setattr(name, value)

    @property
    def injection_points(self):
        return {k: attrs.get('value', _NotInitialized) for k, attrs in self._dep_graph.nodes.items() 
                if self._dep_graph.out_degree(k) == 0}

    def _recursive_init(self, key, object_dict:dict, init_args:dict):
        if key not in object_dict and key in self._dep_graph:
            arg_maps = {}
            
            for arg in self._dep_graph.neighbors(key):
                arg_object = _NotInitialized

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
                    if not self.strict:
                        print(error_message)
                        break
                    else:
                        raise KeyError(error_message)

                arg_maps[arg] = arg_object

            
            arg_len = self._dep_graph.out_degree(key)
            if len(arg_maps) == arg_len:
                object_dict[key] = self._dep_graph.nodes[key]['initiator'](**arg_maps)

    def __setattr__(self, key, value):
        if callable(value):
            self._dep_graph.add_node(key, initiator=value)
            
            argspecs = inspect.getfullargspec(value)
            for a in argspecs.args:
                if a != 'self':
                    self._dep_graph.add_node(a)
                    self._dep_graph.add_edge(key, a)

            defaults = argspecs.defaults or []
            for name, value in zip(argspecs.args[::-1], defaults[::-1]):
                self._dep_graph.add_node(name, value=value)

        return super().__setattr__(key, value)

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
