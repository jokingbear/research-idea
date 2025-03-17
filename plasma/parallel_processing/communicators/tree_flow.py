import pandas as pd
import networkx as nx

from ...functional import State, partials
from ..queues import Queue


class TreeFlow(State):

    def __init__(self):
        super().__init__()

        self._module_graph = nx.DiGraph()

    def register_chain(self, queue:Queue, block1=None, block2=None):
        block1 = block1 or _ProxyIO
        block2 = block2 or _ProxyIO
        assert callable(block1) or block1 is _ProxyIO, 'block1 must be callable'
        assert callable(block2) or block2 is _ProxyIO, 'block2 must be callable'
        assert block1 is not _ProxyIO or block2 is not _ProxyIO, 'one of the two block must not be empty'
        assert block2 not in self._module_graph or block2 in self._module_graph \
            and block2 is not _ProxyIO \
            and len([*self._module_graph.predecessors(block2)]) == 0, 'block2 is already registered'

        self._module_graph.add_edge(block1, block2, queue=queue)

    def inputs(self):
        results = []
        for n in self._module_graph.successors(_ProxyIO):
            q = self._module_graph.edges[_ProxyIO, n]['queue']
            results.append((q, n))
        return results

    def outputs(self):
        results = []
        for n in self._module_graph.predecessors(_ProxyIO):
            q = self._module_graph.edges[n, None]
            if q is not None:
                results.append((n, q))
    
    def run(self):
        data_graph = self._build_data_graph()

        for q in data_graph:
            q:Queue
            successors:list[Queue] = [n for n in data_graph.successors(q) if n is not None]
            if len(successors) > 0:
                block = data_graph.edges[q, successors[0]]['block']
                q.register_callback(block)\
                    .chain(partials(_propagate, successors))\
                    .run()
            elif 'block' in data_graph.nodes[q]:
                block = data_graph.nodes[q]['block']
                q.register_callback(block).run()
        
        return self

    def _build_data_graph(self):
        graph = nx.DiGraph()

        for b in self._module_graph:
            if b is not _ProxyIO:
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

    def __repr__(self):
        flows = []
        for n in self._module_graph.successors(_ProxyIO):
            flows.append(_render(self._module_graph, n))
        flows = ('\n' + '=' * 100 + '\n').join(flows)
        return flows

    def __enter__(self):
        return self.run()
    
    def release(self):
        for edge_attrs in self._module_graph.edges.values():
            queue:Queue = edge_attrs['queue']
            queue.release()
    
    def __exit__(self, *_):
        self.release()


def _propagate(queues:list[Queue], x):
    for q in queues:
        q.put(x)


class _ProxyIO:
    pass


def _render(graph:nx.DiGraph, key, indent='\t'):
    if key is not _ProxyIO:
        lines = [str(key)]
        for n in graph.successors(key):
            rendered_lines = _render(graph, n, indent)
            rendered_lines = rendered_lines.split('\n')
            rendered_lines[0] = '|--' + rendered_lines[0]
            rendered_lines = [indent + l for l in rendered_lines]
            lines.extend(rendered_lines)
        lines = '\n'.join(lines)
        return lines
    return ''
