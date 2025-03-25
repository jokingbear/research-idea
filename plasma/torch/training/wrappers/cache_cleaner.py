import torch

from ..bases import ForwardWrapper


class CacheCleaner(ForwardWrapper):

    def __init__(self, clean_step):
        super().__init__()

        self.clean_step = clean_step
        self._counter = 0

    def prepend(self, trainer, i, inputs):
        if self._counter % self.clean_step == 0:
            torch.cuda.empty_cache()
        
        self._counter += 1
