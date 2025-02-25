import torch

from ....functional import SimplePipe
from .trainer import Trainer
from functools import wraps


class TrainerWrapper(SimplePipe[type[Trainer], type[Trainer]]):
    
    def forward(self, trainer:Trainer, i:int, inputs, outputs:torch.Tensor):
        return outputs
    
    def backward(self, trainer:Trainer, i:int, inputs, obj_val: torch.Tensor):
        pass
    
    def run(self, trainer_class:type[Trainer]):
        trainer_class.forward = self._chain_forward(trainer_class.forward)
        trainer_class.backward = self._chain_backward(trainer_class.backward)
        return trainer_class
    
    def _chain_forward(self, forwarder):
        @wraps(forwarder)
        def alt_forward(trainer, i, inputs):
            outputs = forwarder(trainer, i, inputs)

            if outputs is not None:
                self.forward(trainer, i, inputs, outputs)
                return outputs
        
        return alt_forward

    def _chain_backward(self, backwarder):
        @wraps(backwarder)
        def alt_backward(trainer, i, inputs, obj_val):
            backwarder(trainer, i, inputs, obj_val)
            self.backward(trainer, i, inputs, obj_val)
        
        return alt_backward
