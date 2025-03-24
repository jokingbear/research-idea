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

    def register_chains(self, *chains:tuple[Queue, str, str]):
        for queue, block1, block2 in chains:
            assert block1 is None or hasattr(self, block1), 'block1 must be an attribute of the flow or None'
            assert block2 is None or hasattr(self, block2), 'block2 must be an attribute of the flow or None '
            assert block1 is not None or block2 is not None, 'one of the two block must not be empty'
            assert block2 is None or block2 not in self._module_graph or (block1, block2) in self._module_graph.edges, \
                'block2 already has a predecessor'
            
            assert block1 is not None or (block1 is None and \
                                          (ProxyIO not in self._module_graph or self._module_graph.out_degree(ProxyIO) < 1)), \
                    'TreeFlow only allows one input block'

            block1 = block1 or ProxyIO
            block2 = block2 or ProxyIO
            self._module_graph.add_edge(block1, block2, queue=queue)

    @property
    def inputs(self)->dict[str, Queue]:
        results = {}
        for n in self._module_graph.successors(ProxyIO):
            q = self._module_graph.edges[ProxyIO, n]['queue']
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
        data_graph = self._build_data_graph()

        for q in data_graph:
            q:Queue
            successors:list[Queue] = [n for n in data_graph.successors(q) if n is not None]
            if len(successors) > 0:
                attr_name = data_graph.edges[q, successors[0]]['block']
                block = getattr(self, attr_name)
                if not isinstance(block, Distributor):
                    block = UniformDistributor(block)
                runner = partials(block, *successors, pre_apply_before=False)
                q.register_callback(runner).run()
            elif 'block' in data_graph.nodes[q]:
                attr_name = data_graph.nodes[q]['block']
                block = getattr(self, attr_name)
                q.register_callback(block).run()
        
        return self

    def _build_data_graph(self):
        graph = nx.DiGraph()

        for b in self._module_graph:
            if b is not ProxyIO:
                b0 = [*self._module_graph.predecessors(b)][0]
                q0 = self._module_graph.edges[b0, b]['queue']

                successors = [*self._module_graph.successors(b)]
                if len(successors) == 0:
                    graph.add_node(q0, block=b)
                else:
                    for b1 in successors:
                        q1 = self._module_graph.edges[b, b1]['queue']
                        graph.add_edge(q0, q1, block=b)
        
        return graph

    def __setattr__(self, key: str, value):
        if key[0] != '_':
            assert not isinstance(value, (Queue,)), 'cannot assign a queue as a block'
            assert callable(value), 'public attribute must be callable'

        return super().__setattr__(key, value)

    def __repr__(self):
        flows = []
        for n in self._module_graph.successors(ProxyIO):
            flow_lines = [_render_queue(self._module_graph.edges[ProxyIO, n]['queue'])]
            lines = self._render_lines(n)
            lines[0] = '|-' + lines[0]
            indent = ' ' * 2
            lines = [indent + l for l in lines]
            flow_lines.extend(lines)
            flows.append('\n\n'.join(flow_lines))

        flows = ('\n' + '=' * 100 + '\n').join(flows)
        return flows

    def __enter__(self):
        return self.run()
    
    def release(self):
        for (s, e), edge_attrs in self._module_graph.edges.items():
            if e is not ProxyIO:
                queue:Queue = edge_attrs['queue']
                queue.release()
    
    def __exit__(self, *_):
        self.release()
    
    def _render_lines(self, key):
        if key is not ProxyIO:
            distributor = block = getattr(self, key)
            if isinstance(distributor, Distributor):
                block = distributor.block

            if type(block).__name__ == 'function' or isinstance(block, proxy_func):
                name = repr(block)
            else:
                name = type(block).__name__

            process_txt = ''
            if isinstance(distributor, Distributor):
                process_txt = f'-{type(distributor).__name__}'

            lines = [f'({key}:{name}){process_txt}']
            for n in self._module_graph.successors(key):
                indent = ' ' * 2
                q:Queue = self._module_graph.edges[key, n]['queue']
                qtext = _render_queue(q)
                lines.append(f'{indent}|-{qtext}')
                indent += indent

                rendered_lines = self._render_lines(n)
                if len(rendered_lines) > 0:
                    rendered_lines[0] = '|-' + rendered_lines[0]
                    rendered_lines = [indent + l for l in rendered_lines]
                    lines.extend(rendered_lines)
            return lines
        return []


def _render_queue(queue:Queue):
    num_runner = f'(runner={queue.num_runner})'
    return f'[{type(queue).__name__}{num_runner}]'
