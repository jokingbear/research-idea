from ....functional import AutoPipe
from ...queues import Queue
from abc import abstractmethod


class Processor(AutoPipe):

    def __init__(self, block):
        super().__init__()

        self.block = block

    def run(self, data, *queues:Queue):
        results = self.block(data)
        self.resolve_outputs(results, *queues)

    @abstractmethod
    def resolve_outputs(self, data, *queues:Queue):
        pass
    
    def __repr__(self):
        return f'{type(self.block)}-{type(self)}'
