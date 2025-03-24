import torch

from ..bases import ForwardWrapper


class CacheCleaner(ForwardWrapper):

    def __init__(self, clean_step):
        super().__init__()

        self.clean_step = clean_step

    def append(self, trainer, i, inputs, outputs):
        if i == self.clean_step:
            torch.cuda.empty_cache()
