import torch

from .trainer import Trainer
from functools import wraps


class BackwardWrapper:
    
    def prepend(self, trainer:Trainer, i:int, inputs, objective_val:torch.Tensor):
        pass
    
    def append(self, trainer:Trainer, i:int, inputs, objective_val:torch.Tensor):
        pass
    
    def combine(self, trainer:Trainer, backward_func, i:int, inputs, objective_val:torch.Tensor):
        self.prepend(trainer, i, inputs, objective_val)
        backward_func(trainer, i, inputs, objective_val)
        self.append(trainer, i, inputs, objective_val)

    def __call__(self, backward_func):

        @wraps(backward_func)
        def alt_backward(trainer, i, inputs, obj_val):
            return self.combine(trainer, backward_func, i, inputs, obj_val)

        return alt_backward
