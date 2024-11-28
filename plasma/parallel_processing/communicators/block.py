from ...functional import AutoPipe
from ..queues import QueuePrototype

from abc import abstractmethod


class BlockPrototype(AutoPipe):

    def run(self):
        assert hasattr(self, '_inputs') and self._inputs is not None, 'register_inputs has not been called on his block'
        self._inputs.run()

    def register_inputs(self, queue:QueuePrototype):
        if queue.running:
            queue.release()
        
        queue.register_callback(self.on_received)
        self._inputs = queue
    
    def register_outputs(self, queue:QueuePrototype):
        assert hasattr(self, '_inputs') and self._inputs is not None, 'register_inputs has not been called on his block'
        
        if hasattr(self, '_outputs') and self._outputs is not None:
            self.register_inputs(self._inputs)

        self._inputs.chain(queue.put)
        self._outputs = queue

    @abstractmethod
    def on_received(self, data):
        pass

    def release(self):
        self._inputs.release()
