from .base import Distributor


class IteratorDistributor(Distributor):
    
    def resolve_outputs(self, data, *queues):        
        for r in data:
            for q in queues:
                q.put(r)
