from .base_block import Block


class ChainedBlock(Block):

    def __init__(self, block1, create_block_func, *args, **kwargs):
        assert isinstance(block1, Block), f'block1 must be an instance of {Block.__qualname__}'
        block2 = create_block_func(*args, **kwargs)

        assert isinstance(block2, Block), f'block2 must return an instance of {Block.__qualname__}'
        super().__init__(block1.inputs, block2.outputs)
        
        self._prev_block = block1
        self._next_block = block2
    
    def init(self):
        self._prev_block.init()
        self._next_block.init()

    def terminate(self, exc_type, exc_val, exc_tb):
        self._next_block.terminate(exc_type, exc_val, exc_tb)
        self._prev_block.terminate(exc_type, exc_val, exc_tb)
