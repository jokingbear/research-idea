import torch

from ....functional import AutoPipe
from abc import abstractmethod
from .trainer import BaseTrainer


class TrainerWrapper(AutoPipe):
    
    @abstractmethod
    def chain(self, trainer:BaseTrainer, i:int, inputs, outputs:torch.Tensor):
        pass

    def run(self, trainer_class:type[BaseTrainer]):
        trainer_class.forward = self.combine_forward(trainer_class.forward)
        return trainer_class
    
    def combine_forward(self, forwarder):
        def alt_forward(trainer, i, inputs):
            outputs = forwarder(trainer, i, inputs)
            self.chain(trainer, i, inputs, outputs)
            return outputs
        
        return alt_forward
