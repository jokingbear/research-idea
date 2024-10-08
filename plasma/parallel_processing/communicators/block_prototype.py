from ...functional import AutoPipe
from ..queues import QueuePrototype


class BlockPrototype(AutoPipe):

    def __init__(self, in_queue:QueuePrototype, out_queue:QueuePrototype):
        super().__init__()

        in_queue.chain(out_queue.put)
        self.inputs = in_queue
        self.outputs = out_queue

    def run(self):
        self.outputs.run()
    
    def put(self, data):
        self.inputs.put(data)

    def release(self):
        self.inputs.release()
        self.outputs.release()
