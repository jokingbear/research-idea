import pandas as pd
import networkx as nx

from ...functional import State, partials, proxy_func
from ..queues import Queue
from .distributors import Distributor, UniformDistributor
from ._proxy import ProxyIO


class TreeFlow(State):

    def __init__(self):
        super().__init__()

        self._module_graph = nx.DiGraph()

    def chain(self, *blocks:tuple[str, str]|tuple[str, str, Queue]|tuple[str, str, Queue, Distributor]):
        for block1, block2, *block2_params in blocks:                
            assert block1 is None or hasattr(self, block1), 'block1 must be an attribute of the flow or None'
            assert block2 is None or hasattr(self, block2), 'block2 must be an attribute of the flow or None'
            assert block1 is not None or block2 is not None, 'one of the two block must not be empty'
            
            assert block1 is not None or ProxyIO not in self._module_graph \
                or (ProxyIO, block2) in self._module_graph.edges, \
                    'TreeFlow only allows one input block'

            assert block2 is not None or len(block2_params) == 1, 'chain outputs must be of the form str, None, queue'

            block1 = block1 or ProxyIO
            block2 = block2 or ProxyIO
            
            if len(block2_params) == 1:
                block2_params.append(UniformDistributor())

            params = {}
            for i, p in enumerate(block2_params):
                if i == 0:
                    params['queue'] = p
                else:
                    params['dist'] = p

            self._module_graph.add_node(block1)
            if block2 is ProxyIO:
                self._module_graph.add_edge(block1, block2, **params)
            else:
                self._module_graph.add_node(block2, **params)
                self._module_graph.add_edge(block1, block2)

        return self

    @property
    def inputs(self)->dict[str, Queue]:
        results = {}
        for n in self._module_graph.successors(ProxyIO):
            q = self._module_graph.nodes[n]['queue']
            results[n] = q
        return results

    @property
    def outputs(self)->dict[str, Queue]:
        results = {}
        for n in self._module_graph.predecessors(ProxyIO):
            q = self._module_graph.edges[n, ProxyIO]['queue']
            results[n] = q
        return results

    def run(self):
        assert ProxyIO in self._module_graph and self._module_graph.out_degree(ProxyIO) > 0, 'flow does not have an input'

        for b in self._module_graph:
            if b is not ProxyIO:
                block = getattr(self, b)
                distributor:Distributor = self._module_graph.nodes[b]['dist']
                q:Queue = self._module_graph.nodes[b]['queue']
                next_qs = []
                for next_b in self._module_graph.successors(b):
                    if next_b is ProxyIO:
                        next_qs.append(self._module_graph.edges[b, next_b]['queue'])
                    else:
                        next_qs.append(self._module_graph.nodes[next_b]['queue'])
                q.register_callback(block)\
                    .chain(partials(distributor, *next_qs, pre_apply_before=False))\
                        .run()
        
        return self

    def __setattr__(self, key: str, value):
        if key[0] != '_':
            assert not isinstance(value, (Queue,)), 'cannot assign a queue as a block'
            assert callable(value), 'public attribute must be callable'

        return super().__setattr__(key, value)

    def __repr__(self):     
        for n in self._module_graph.successors(ProxyIO):
            rendered = set()
            flow_lines = self._render_lines(n, rendered)
            flow_lines[0] = '*-' + flow_lines[0]
            return '\n\n'.join(flow_lines)

    def __enter__(self):
        return self.run()
    
    def release(self):
        for b, attrs in self._module_graph.nodes.items():
            if b is not ProxyIO:
                queue:Queue = attrs['queue']
                queue.release()
    
    def __exit__(self, *_):
        self.release()
    
    def _render_lines(self, key, rendered:set):
        lines = []
        if key is not ProxyIO:
            block = getattr(self, key)
            distributor = self._module_graph.nodes[key]['dist']
            queue:Queue = self._module_graph.nodes[key]['queue']

            if type(block).__name__ == 'function' or isinstance(block, proxy_func):
                name = repr(block)
            else:
                name = type(block).__name__

            process_txt = ''
            if not isinstance(distributor, UniformDistributor):
                process_txt = f'-{type(distributor).__name__}'
            
            initial_indent = ' ' * 2
            lines = [
                f'[{type(queue).__name__}(runner={queue.num_runner})]',
                f'{initial_indent}|-({key}:{name}){process_txt}'
            ]
            if key not in rendered:
                for n in self._module_graph.successors(key):
                    indent = initial_indent * 2
                    if n is ProxyIO:
                        queue = self._module_graph.edges[key, n]['queue']
                        lines.append(f'{indent}|-[{type(queue).__name__}(runner={queue.num_runner})]-*')
                    else:
                        rendered_lines = self._render_lines(n, rendered)
                        if len(rendered_lines) > 0:
                            rendered_lines[0] = '|-' + rendered_lines[0]
                            rendered_lines = [indent + l for l in rendered_lines]
                            lines.extend(rendered_lines)
            else:
                lines[-1] += '...'
        rendered.add(key)
        return lines 
