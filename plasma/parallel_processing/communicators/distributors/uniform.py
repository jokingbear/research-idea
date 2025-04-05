from .base import Distributor


class UniformDistributor(Distributor):

    def run(self, data, *queues, **named_queues):
        for q in queues:
            q.put(data)
        
        for q in named_queues.values():
            q.put(data)
