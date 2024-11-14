from ...functional import AutoPipe
from ..queues import QueuePrototype
from abc import abstractmethod
from .block import BlockPrototype


class CallerPrototype(AutoPipe):

    def __init__(self, in_queue:QueuePrototype, out_queue:QueuePrototype):
        super().__init__()
        
        assert not out_queue.running, 'out queue should only be run inside this caller'
        out_queue.register_callback(self.on_received)
        self.inputs = in_queue
        self.outputs = out_queue
        self._blocks:list[BlockPrototype] = []

    @abstractmethod
    def on_received(self, data):
        pass

    def run(self):
        [b.run() for b in self._blocks]
        if not self.outputs.running:
            self.outputs.run()
    
    def put(self, data):
        self.inputs.put(data)

    def release(self):
        [b.release() for b in self._blocks]
        self.outputs.release()

    def __setattr__(self, key: str, value):
        if isinstance(value, BlockPrototype):
            self._blocks.append(value)
        
        return super().__setattr__(key, value)
