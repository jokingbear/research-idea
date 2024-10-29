from ....functional import AutoPipe, chain
from abc import abstractmethod
from .trainer import BaseTrainer


class TrainerWrapper(AutoPipe):
    
    @abstractmethod
    def chain(self, trainer, state, i, inputs, outputs):
        pass

    def run(self, trainer_class:type[BaseTrainer]):
        trainer_class.forward = self.combine_forward(trainer_class.forward)
        
        return trainer_class
    
    def combine_forward(self, forwarder):
        def alt_forward(trainer, state, i, inputs):
            outputs = forwarder(trainer, state, i, inputs)
            self.chain(trainer, state, i, inputs, outputs)
            return outputs
        
        return alt_forward
