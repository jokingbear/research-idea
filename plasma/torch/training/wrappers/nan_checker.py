import torch

from ..prototypes.trainer_wrapper import TrainerWrapper


class NanChecker(TrainerWrapper):

    def chain(self, trainer, state, i, inputs, outputs):
        if bool(torch.isnan(outputs).prod()):
            raise RuntimeError('loss is nan')
