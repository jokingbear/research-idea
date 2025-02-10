import torch

from ..bases.trainer_wrapper import TrainerWrapper


class NanChecker(TrainerWrapper):

    def forward(self, trainer, i, inputs, outputs):
        if bool(torch.isnan(outputs).prod()):
            raise RuntimeError('loss is nan')
