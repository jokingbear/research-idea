from .base import Distributor


class UniformDistributor(Distributor):

    def run(self, data, *queues):
        for q in queues:
            q.put(data)
