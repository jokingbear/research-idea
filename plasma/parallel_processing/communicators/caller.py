from ...functional import AutoPipe
from ..queues import QueuePrototype
from abc import abstractmethod
from .block import BlockPrototype


class CallerPrototype(AutoPipe):

    def __init__(self,):
        super().__init__()
        self._blocks:dict[str, BlockPrototype] = {}

    @abstractmethod
    def on_received(self, data):
        pass

    def run(self):
        assert hasattr(self, '_inputs') and hasattr(self, '_outputs'), 'register_inout has not been called on this instance'
        assert len(self._blocks) > 0, 'Caller must have at least 1 block'

        [b.run() for b in self._blocks.values()]
        if not self._outputs.running:
            self._outputs.run()
    
    def register_inout(self, outputs:QueuePrototype, **block_inputs:QueuePrototype):
        last_block:BlockPrototype = None
        for i, (k, current_block) in enumerate(self._blocks.items()):
            assert k in block_inputs, f'block {k} is not in input dict'

            if i == 0:
                self._inputs = block_inputs[k]
            
            pipe = block_inputs[k]
            current_block.register_inputs(pipe)
            if last_block is not None:
                last_block.register_outputs(pipe)

            last_block = current_block

        current_block.register_outputs(outputs)
        outputs.register_callback(self.on_received)
        self._outputs = outputs

    def put(self, data):
        assert hasattr(self, '_inputs'), 'register_inout method has not been called on this caller'
        self._inputs.put(data)

    def release(self):
        assert hasattr(self, '_outputs'), 'register_inout method has not been called on this caller'
        [b.release() for b in self._blocks.values()]
        self._outputs.release()

    def __setattr__(self, key: str, value):
        if isinstance(value, BlockPrototype):
            self._blocks[key] = value
        
        return super().__setattr__(key, value)
