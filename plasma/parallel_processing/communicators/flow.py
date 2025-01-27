import pandas as pd

from ...functional import State, LambdaPipe, AutoPipe
from ..queues import Queue


class Flow(State):

    def __init__(self):
        super().__init__()

        self._pipes:dict[str, AutoPipe] = {}
        self._blocks:list[Queue|Flow] = []

    def run(self):
        assert hasattr(self, 'inputs') and hasattr(self, 'outputs'), 'registerIOs has not been called on this instance.'
        assert len(self._pipes) > 0, 'Flow must have at least 1 block.'

        [b.run() for b in self._blocks]
        return self
    
    def registerIOs(self, **pipeIOs:Queue|dict[str, Queue]):
        blocks = pd.Series(self._pipes)
        for i, current_block in enumerate(blocks):            
            if isinstance(current_block, Flow):
                inputs, outputs = self._resolve_flow(i, blocks, pipeIOs)
                self._blocks.append(current_block)
            else:
                inputs, outputs = self._resolve_pipe(i, blocks, pipeIOs)
                self._blocks.append(inputs)
            
            if i == 0:
                self.inputs = inputs
        
        self.outputs = outputs

    def _resolve_flow(self, index:int, blocks:pd.Series, pipeIOs:dict) -> tuple[Queue, Queue]:
        block_key = blocks.index[index]

        input_queues = pipeIOs[block_key]
        assert isinstance(input_queues, dict), f'{block_key} must be of type dict'
        outputs = self._resolve_outputs(index, blocks, pipeIOs)

        block:Flow = blocks.loc[block_key]
        block.registerIOs(**input_queues, outputs=outputs)
        
        return block.inputs, block.outputs

    def _resolve_pipe(self, index:int, blocks:pd.Series, pipeIOs:dict) -> tuple[Queue, Queue]:
        block_key = blocks.index[index]

        inputs = pipeIOs[block_key]
        assert isinstance(inputs, Queue), f'{block_key} must be of type QueuePrototype'

        pipe = blocks.iloc[index]
        inputs.register_callback(pipe)
        outputs = self._resolve_outputs(index, blocks, pipeIOs)
        if outputs is not None:
            inputs.chain(outputs.put)
        
        return inputs, outputs

    def _resolve_outputs(self, index:int, blocks:pd.Series, pipeIOs:dict) -> Queue:
        next_index = index + 1
        outputs = None
        current_key = blocks.index[index]
        if f'{current_key}_outputs' in pipeIOs:
            outputs = pipeIOs[f'{current_key}_outputs']
            ref = f'{current_key}_outputs'
        elif next_index < len(blocks):
            next_key = blocks.index[next_index]
            outputs = pipeIOs[next_key]
            if isinstance(outputs, dict):
                for k, q in outputs.items():
                    ref = f'{next_key}.{k}'
                    outputs = q
                    break
        elif next_index == len(blocks) and 'outputs' in pipeIOs:
            outputs = pipeIOs['outputs']
            ref = 'outputs'

        assert isinstance(outputs, Queue) or outputs is None, f'{ref} must be of type QueuePrototype'
        return outputs

    def put(self, data):
        assert hasattr(self, 'inputs'), 'register_inout method has not been called on this caller'
        self.inputs.put(data)

    def release(self):
        assert hasattr(self, 'outputs'), 'register_inout method has not been called on this caller'
        [b.release() for b in self._blocks]
        [b.release() for b in self._pipes.values() if isinstance(b, State)]

        if self.outputs is not None:
            self.outputs.release()

    def __setattr__(self, key: str, value):
        if key[0] != '_' and key not in {'inputs', 'outputs'}:
            assert not isinstance(value, Queue), 'cannot assign a queue as a block'

            if isinstance(value, AutoPipe):
                self._pipes[key] = value
            elif callable(value):
                self._pipes[key] = LambdaPipe(value)
            else:
                raise ValueError(f'{key} is not an Autopipe or function instance.')
            
        return super().__setattr__(key, value)

    def __repr__(self):
        texts = []
        for k, v in self._pipes.items():
            text_v = repr(v) if isinstance(v, LambdaPipe) else type(v)
            texts.append(f'{k}-{text_v}')
            if isinstance(v, Flow):
                v_repr = repr(v)
                texts.extend('\t' + s for s in v_repr.split('\n'))
        return '\n'.join(texts)

    def __enter__(self):
        return self.run()
    
    def __exit__(self, *_):
        self.release()
