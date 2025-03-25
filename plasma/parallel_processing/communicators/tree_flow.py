import pandas as pd
import networkx as nx

from ...functional import State, partials, proxy_func
from ..queues import Queue
from .distributors import Distributor, UniformDistributor
from ._proxy import ProxyIO
from ...logging import ExceptionLogger


class TreeFlow(State):

    def __init__(self):
        super().__init__()

        self._module_graph = nx.DiGraph()

    def register_chains(self, *chains:tuple[str, Queue, str]):
        for block1, queue, block2 in chains:
            assert block1 is None or hasattr(self, block1), 'block1 must be an attribute of the flow or None'
            assert block2 is None or hasattr(self, block2), 'block2 must be an attribute of the flow or None'
            assert isinstance(queue, Queue), f'queue must be an instance of {Queue}'
            assert block1 is not None or block2 is not None, 'one of the two block must not be empty'
            
            assert block1 is not None or (block1 is None and \
                                          (ProxyIO not in self._module_graph or self._module_graph.out_degree(ProxyIO) < 1)), \
                    'TreeFlow only allows one input block'

            block1 = block1 or ProxyIO
            block2 = block2 or ProxyIO
            self._module_graph.add_node(block1)
            if block2 is ProxyIO:
                self._module_graph.add_edge(block1, block2, queue=queue)
            else:
                self._module_graph.add_node(block2, queue=queue, dist=UniformDistributor())
                self._module_graph.add_edge(block1, block2)

    def register_distributors(self, *block:tuple[str, Distributor]):
        for b, dist in block:
            self._module_graph.add_node(b, dist=dist)

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
        flows = []
        rendered = set()
        for n in self._module_graph.successors(ProxyIO):
            flow_lines = self._render_lines(n, rendered)
            flows.append('\n\n'.join(flow_lines))

        flows = ('\n' + '=' * 100 + '\n').join(flows)
        return flows

    def __enter__(self):
        return self.run()
    
    def release(self):
        for b, attrs in self._module_graph.nodes.items():
            if b is not ProxyIO:
                queue:Queue = attrs['queue']
                queue.release()
    
    def __exit__(self, *_):
        self.release()
    
    @ExceptionLogger(log_func=lambda exio: print(f'{exio.args[1]}\n{exio.exception}'))
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
                        lines.append(f'{indent}|-[{type(queue).__name__}(runner={queue.num_runner})]')
                    else:
                        rendered_lines = self._render_lines(n, rendered)
                        if len(rendered_lines) > 0:
                            rendered_lines[0] = '|-' + rendered_lines[0]
                            rendered_lines = [indent + l for l in rendered_lines]
                            lines.extend(rendered_lines)
                    
        rendered.add(key)
        return lines 
