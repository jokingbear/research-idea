import torch

from .trainer import Trainer
from functools import wraps


class ForwardWrapper:
    
    def prepend(self,  trainer:Trainer, i:int, inputs):
        pass
    
    def append(self, trainer:Trainer, i:int, inputs, outputs:torch.Tensor):
        pass
    
    def combine(self, trainer:Trainer, forward_func, i:int, inputs):
        self.prepend(trainer, i, inputs)
        outputs = forward_func(trainer, i, inputs)
        self.append(trainer, i, inputs, outputs)
        return outputs

    def __call__(self, forward_func):
        
        @wraps(forward_func)
        def alt_forward(trainer, i, inputs):
            return self.combine(trainer, forward_func, i, inputs)
        
        return alt_forward
