from ..queues import Queue
from .tree2 import StableTree
from warnings import warn
from .distributors import Distributor
from ._proxy import ProxyIO


class Sequential(StableTree):

    def registerIOs(self, **blocks:Queue|tuple[Queue, Distributor]):
        chains = []
        for key, value in blocks.items():  
            if not isinstance(value, tuple):
                value = (value,)

            if key == 'outputs':
                prev_block = self._marked_attributes[-1]
                triplets = prev_block, None, *value
            else:
                attr_idx = [i for i, attr in enumerate(self._marked_attributes) if attr == key][0]
                if attr_idx == 0:
                    prev_block = None
                else:
                    prev_block = self._marked_attributes[attr_idx - 1]
                triplets = prev_block, key, *value 
            
            chains.append(triplets)

        self.chain(*chains)
        return self


    def __setattr__(self, key, value):
        assert key not in {'outputs'}, 'outputs is reserved'

        if key[0] != '_' and ProxyIO in self._module_graph and self._module_graph.in_degree(ProxyIO) > 0:
            print('adding new block after outputs already registered, removed old outputs')

            predecessors = [*self._module_graph.predecessors(ProxyIO)]
            for n in predecessors:
                self._module_graph.remove_edge(n, ProxyIO)

        return super().__setattr__(key, value)
