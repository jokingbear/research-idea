import torch
import torch.amp as amp

from ....functional import partials
from ..prototypes.trainer_wrapper import TrainerWrapper


class AMP(TrainerWrapper):

    def __init__(self, cast_type=torch.bfloat16, unscale=True):
        super().__init__()

        self._scaler = amp.GradScaler()
        self.cast_type = cast_type
        self.unscale = unscale

    def run(self, trainer_class):
        new_trainer_class = super().run(trainer_class)
        new_trainer_class.backward = partials(self.backward, self)
        return new_trainer_class
    
    def combine_forward(self, forwarder):
        def alt_forward(trainer, i, inputs):
            with torch.autocast('cuda', self.cast_type):
                outputs = forwarder(trainer, i, inputs)

            self.chain(trainer, i, inputs, outputs)
            return outputs

        return alt_forward

    def chain(self, trainer, i, inputs, outputs):
        self._scaler.scale(outputs).backward()
        if self.unscale:
            opt = trainer._optimizer
            self._scaler.unscale_(opt)
    
    def backward(self, trainer, obj_val):
        opt = trainer.optimizer
        self._scaler.step(opt)
        self._scaler.update()
