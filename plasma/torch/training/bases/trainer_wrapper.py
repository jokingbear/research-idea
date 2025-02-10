import torch

from ....functional import SimplePipe
from abc import abstractmethod
from .trainer import Trainer


class TrainerWrapper(SimplePipe[type[Trainer], type[Trainer]]):
    
    @abstractmethod
    def chain(self, trainer:Trainer, i:int, inputs, outputs:torch.Tensor):
        pass

    def run(self, trainer_class:type[Trainer]):
        trainer_class.forward = self.combine_forward(trainer_class.forward)
        return trainer_class
    
    def combine_forward(self, forwarder):
        def alt_forward(trainer, i, inputs):
            outputs = forwarder(trainer, i, inputs)

            if outputs is not None:
                self.chain(trainer, i, inputs, outputs)
                return outputs
        
        return alt_forward
