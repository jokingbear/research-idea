from ...functional import AutoPipe
from ..queues import QueuePrototype
from abc import abstractmethod


class CallerPrototype(AutoPipe):

    def __init__(self, in_queue:QueuePrototype, out_queue:QueuePrototype):
        super().__init__()
        
        assert not out_queue.running, 'out queue should only be run inside this caller'
        out_queue.register_callback(self.on_received)
        self.inputs = in_queue
        self.outputs = out_queue

    @abstractmethod
    def on_received(self, data):
        pass

    def register_callback(self, callback):
        self.outputs.register_callback(callback)

    def run(self):
        self.outputs.run()
    
    def put(self, data):
        self.inputs.put(data)

    def release(self):
        self.outputs.release()
