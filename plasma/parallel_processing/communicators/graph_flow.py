import pandas as pd
import networkx as nx

from ...functional import State, LambdaPipe
from ..queues import Queue


class GraphFlow(State):

    def __init__(self):
        super().__init__()

        self._module_graph = nx.DiGraph()
        self._data_graph = nx.DiGraph()

    def registerIOs(self, pipeIOs:dict[tuple[str, str], Queue]):
        input_index = 0
        output_index = 0
        for (n1, n2), q in pipeIOs.items():
            assert n1 in self._marked_attributes, f'{n1} is not assigned'
            assert n2 in self._marked_attributes, f'{n2} is not assigned'

            if n1 is None:
                n1 = f'input-{input_index}'
                input_index += 1
            elif n2 is None:
                n2 = f'output-{output_index}'
                output_index += 1

            self._module_graph.add_edge(n1, n2, queue=q)
    
    def inputs(self):
        inputs = [n for n in self._module_graph if self._module_graph.in_degree(n) == 0]
        return inputs

    def outputs(self):
        outputs = [n for n in self._module_graph if self._module_graph.out_degree(n) == 0]
        return outputs

    def run(self):
        return self

    def _build_queue_graph(self):
        graph = nx.DiGraph()
        
        for (n1, n2) in graph.edges:
            pass

    def put(self, data):
        assert hasattr(self, 'inputs'), 'register_inout method has not been called on this caller'
        self.inputs.put(data)

    def release(self):
        assert hasattr(self, 'outputs'), 'register_inout method has not been called on this caller'
        [b.release() for b in self._blocks]
        [b.release() for b in self._pipes.values() if isinstance(b, State)]

        if self.outputs is not None:
            self.outputs.release()

    def __setattr__(self, key: str, value):
        if key[0] != '_':
            assert not isinstance(value, Queue), 'cannot assign a queue as a block'

            if callable(value):
                self._module_graph.add_node(key, func=value, label=type(value))
            else:
                raise ValueError(f'{key} is not an Autopipe or function instance.')
            
        return super().__setattr__(key, value)

    def __repr__(self):
        texts = []
        for k, v in self._pipes.items():
            text_v = repr(v) if isinstance(v, LambdaPipe) else type(v)
            texts.append(f'{k}-{text_v}')
            if isinstance(v, Flow):
                v_repr = repr(v)
                texts.extend('\t' + s for s in v_repr.split('\n'))
        return '\n'.join(texts)

    def __enter__(self):
        return self.run()
    
    def __exit__(self, *_):
        self.release()
