from ..queues import Queue
from .tree_flow import TreeFlow


class Flow(TreeFlow):

    def registerIOs(self, **pipeIOs:Queue|dict[str, Queue]):
        chains = []
        for key, value in pipeIOs.items():  
            if key == 'outputs':
                prev_block = self._marked_attributes[-1]
                triplets = value, prev_block, None
            else:
                attr_idx = [i for i, attr in enumerate(self._marked_attributes) if attr == key][0]
                if attr_idx == 0:
                    prev_block = None
                else:
                    prev_block = self._marked_attributes[attr_idx - 1]
                triplets = value, prev_block, key
            
            chains.append(triplets)

        self.register_chains(*chains)

    def put(self, x):
        for q in self.inputs.values():
            q.put(x)
