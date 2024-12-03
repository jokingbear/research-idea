from ...functional import State, LambdaPipe, AutoPipe
from ..queues import QueuePrototype


class Flow(State):

    def __init__(self):
        super().__init__()

        self._blocks:dict[str, AutoPipe] = {}
        self._queues:dict[str, QueuePrototype|Flow] = {}

    def run(self):
        assert hasattr(self, 'inputs') and hasattr(self, 'outputs'), 'registerIOs has not been called on this instance.'
        assert len(self._blocks) > 0, 'Flow must have at least 1 block.'

        [b.run() for b in self._queues.values()]
    
    def registerIOs(self, **pipeIOs:QueuePrototype|dict[str, QueuePrototype]):
        blocks = [*self._blocks.items()]
        for i, (k, current_block) in enumerate(blocks):
            inputs = self._resolve_inputs(k, current_block, pipeIOs)
            outputs = self._resolve_outputs(i, k, blocks, pipeIOs)
            
            if isinstance(current_block, Flow):
                current_block.registerIOs(**inputs, outputs=outputs)
            else:
                inputs.register_callback(current_block.run)
                if outputs is not None:
                    inputs.chain(outputs.put)
            
            self._queues[k] = current_block if isinstance(current_block, Flow) else inputs
            if i == 0:
                self.inputs = inputs
        
        self.outputs = outputs

    def _resolve_inputs(self, block_key:str, block:AutoPipe, pipeIOs:dict) -> QueuePrototype|dict[str, QueuePrototype]:
        assert block_key in pipeIOs, f'{block_key} not in pipeIOs'
        inputs = pipeIOs[block_key]
        if isinstance(inputs, dict):
            assert isinstance(block, Flow), f'{block_key} in pipeIOs is a dict, but {block_key} is not a Flow.'
            assert 'outputs' not in inputs, f'{block_key} in pipeIOs cannot contain key outputs.'
        
        return inputs

    def _resolve_outputs(self, current_index:int, current_key:str, 
                         blocks:list[tuple[str, AutoPipe]], 
                         pipeIOs:dict[str, QueuePrototype|dict[str, QueuePrototype]]) -> QueuePrototype:
        next_index = current_index + 1
        outputs = None
        if f'{current_key}_outputs' in pipeIOs:
            outputs = pipeIOs[f'{current_key}_outputs']
        elif next_index < len(blocks):
            next_key, _ = blocks[next_index]
            outputs = pipeIOs[next_key]
            if isinstance(outputs, dict):
                for k, q in outputs.items():
                    assert q is not None, f'{next_key}.{k} cannot be None.'
                    outputs = q
                    break
        elif next_index == len(blocks) and 'outputs' in pipeIOs:
            outputs = pipeIOs['outputs']
        
        return outputs

    def put(self, data):
        assert hasattr(self, 'inputs'), 'register_inout method has not been called on this caller'
        self.inputs.put(data)

    def release(self):
        assert hasattr(self, 'outputs'), 'register_inout method has not been called on this caller'
        [b.release() for b in self._queues.values()]
        [b.release() for b in self._blocks.values() if isinstance(b, State)]

        if self.outputs is not None:
            self.outputs.release()

    def __setattr__(self, key: str, value):
        if key[0] != '_' and key not in {'inputs', 'outputs'}:
            assert not isinstance(value, QueuePrototype), 'cannot assign a queue as a block'

            if isinstance(value, AutoPipe):
                self._blocks[key] = value
            elif callable(value):
                self._blocks[key] = LambdaPipe(value)
            else:
                raise ValueError(f'{key} is not an Autopipe or function instance.')
            
        return super().__setattr__(key, value)

    def __repr__(self):
        texts = []
        for k, v in self._blocks.items():
            text_v = repr(v) if isinstance(v, LambdaPipe) else type(v)
            texts.append(f'{k}-{text_v}')
        return '\n'.join(texts)
