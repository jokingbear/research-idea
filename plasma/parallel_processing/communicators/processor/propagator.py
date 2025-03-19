from .processor import Processor


class Propagator(Processor):

    def resolve_outputs(self, data, *queues):
        for q in queues:
            q.put(data)
