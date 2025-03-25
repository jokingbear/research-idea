from .base import Distributor


class IteratorDistributor(Distributor):
    
    def run(self, data, *queues):        
        for r in data:
            for q in queues:
                q.put(r)
