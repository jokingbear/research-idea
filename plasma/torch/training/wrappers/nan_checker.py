import torch

from ..bases import ForwardWrapper


class NanChecker(ForwardWrapper):

    def append(self, trainer, i, inputs, outputs):
        if bool(torch.isnan(outputs).prod()):
            raise RuntimeError('loss is nan')
