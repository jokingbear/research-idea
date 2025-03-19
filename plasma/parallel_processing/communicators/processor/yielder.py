from .processor import Processor


class Yield(Processor):

    def __init__(self, block):
        super().__init__()

        self.block = block
    
    def resolve_outputs(self, data, *queues):        
        for r in data:
            for q in queues:
                q.put(r)
