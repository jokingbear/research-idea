from .base import Distributor


class IteratorDistributor(Distributor):
    
    def run(self, data, *queues, **named_queues):  
        for r in data:
            for q in queues:
                q.put(r)

            for q in named_queues.values():
                q.put(r)
