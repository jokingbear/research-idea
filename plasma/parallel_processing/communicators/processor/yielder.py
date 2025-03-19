from .processor import Processor


class Yield(Processor):
    
    def resolve_outputs(self, data, *queues):        
        for r in data:
            for q in queues:
                q.put(r)
