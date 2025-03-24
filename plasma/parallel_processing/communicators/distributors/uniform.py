from .base import Distributor


class UniformDistributor(Distributor):

    def resolve_outputs(self, data, *queues):
        for q in queues:
            q.put(data)
