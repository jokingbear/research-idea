import pandas as pd
import networkx as nx

from ...functional import State, partials, proxy_func
from ..queues import Queue
from .processor import Processor, Propagator


class TreeFlow(State):

    def __init__(self):
        super().__init__()

        self._module_graph = nx.DiGraph()

    def register_chains(self, *chains:tuple[Queue, str, str]):
        for queue, block1, block2 in chains:
            assert block1 is None or hasattr(self, block1), 'block1 must be an attribute of the flow or None'
            assert block2 is None or hasattr(self, block2), 'block2 must be an attribute of the flow or None '
            assert block1 is not None or block2 is not None, 'one of the two block must not be empty'
            assert block2 not in self._module_graph or block2 in self._module_graph \
                and block2 is not None \
                and len([*self._module_graph.predecessors(block2)]) == 0, 'block2 already has a predecessor'

            block1 = block1 or _ProxyIO
            block2 = block2 or _ProxyIO
            self._module_graph.add_edge(block1, block2, queue=queue)

    @property
    def inputs(self)->dict[str, Queue]:
        results = {}
        for n in self._module_graph.successors(_ProxyIO):
            q = self._module_graph.edges[_ProxyIO, n]['queue']
            results[n] = q
        return results

    @property
    def outputs(self)->dict[str, Queue]:
        results = {}
        for n in self._module_graph.predecessors(_ProxyIO):
            q = self._module_graph.edges[n, _ProxyIO]['queue']
            results[n] = q
        return results
    
    def run(self):
        data_graph = self._build_data_graph()

        for q in data_graph:
            q:Queue
            successors:list[Queue] = [n for n in data_graph.successors(q) if n is not None]
            if len(successors) > 0:
                attr_name = data_graph.edges[q, successors[0]]['block']
                block:Processor = getattr(self, attr_name)
                runner = partials(block, *successors, pre_apply_before=False)
                q.register_callback(runner).run()
            elif 'block' in data_graph.nodes[q]:
                attr_name = data_graph.nodes[q]['block']
                block:Processor = getattr(self, attr_name)
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

    def __setattr__(self, key: str, value):
        if key[0] != '_':
            assert not isinstance(value, (Queue, TreeFlow)), 'cannot assign a queue/Flow as a block'
            assert callable(value), 'public attribute must be callable'
            if not isinstance(value, Processor):
                value = Propagator(value)

        return super().__setattr__(key, value)

    def __repr__(self):
        flows = []
        for n in self._module_graph.successors(_ProxyIO):
            lines = self._render_lines(n)
            lines[0] = '|-' + lines[0]
            q:Queue = self._module_graph.edges[_ProxyIO, n]['queue']
            queue_text = _render_queue(q)
            indent = ' ' * 2
            lines = [indent + l for l in lines]
            lines.insert(0, queue_text)
            flows.append('\n\n'.join(lines))

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
    
    def _render_lines(self, key):
        if key is not _ProxyIO:
            processor:Processor = getattr(self, key)
            obj = processor.block
            if type(obj).__name__ == 'function' or isinstance(obj, proxy_func):
                name = repr(obj)
            else:
                name = type(obj).__name__

            process_txt = ''
            if not isinstance(processor, Propagator):
                process_txt = f'-{type(processor).__name__}'

            lines = [f'({key}:{name}){process_txt}']
            indent = ' ' * 2
            for n in self._module_graph.successors(key):
                q:Queue = self._module_graph.edges[key, n]['queue']
                qtext = _render_queue(q)
                lines.append(indent + f'|-{qtext}')
                indent += ' ' * 2

                rendered_lines = self._render_lines(n)
                if len(rendered_lines) > 0:
                    rendered_lines[0] = '|-' + rendered_lines[0]
                    rendered_lines = [indent + l for l in rendered_lines]
                    lines.extend(rendered_lines)
            return lines
        return []


class _ProxyIO:
    pass


def _render_queue(queue:Queue):
    num_runner = f'(runner={queue.num_runner})'
    return f'[{type(queue).__name__}{num_runner}]'
