from ...functional import AutoPipe
from ..queues import QueuePrototype

from abc import abstractmethod


class BlockPrototype(AutoPipe):

    def __init__(self, in_queue:QueuePrototype, out_queue:QueuePrototype):
        super().__init__()

        assert not in_queue.running, 'in queue shouldn\'t be run outside'
        in_queue.register_callback(self.on_received)
        in_queue.chain(out_queue.put)

        self.inputs = in_queue
        self.outputs = out_queue

    def run(self):
        self.inputs.run()

    @abstractmethod
    def on_received(self, data):
        pass

    def release(self):
        self.inputs.release()
