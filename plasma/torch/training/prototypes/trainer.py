import torch
import torch.utils
import os

from ....functional import AutoPipe
from collections import namedtuple
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from abc import abstractmethod


TrainerState = namedtuple('TrainerState', ['epoch', 'model', 'optimizer', 'scheduler'])


class BaseTrainer(AutoPipe):
    max_epoch:int

    @abstractmethod
    def init_state(self) -> TrainerState:
        pass

    @abstractmethod
    def init_train_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def forward(self, state:TrainerState, i:int, inputs):
        pass
    
    @abstractmethod
    def backward(self, state:TrainerState, bjective_val:torch.Tensor):
        pass
    

    def finalize_epoch(self, state:TrainerState):
        pass

    def run(self, previous_state=None):
        previous_state = previous_state or self.init_state()
        previous_epoch = previous_state.epoch
        loader = self.init_train_loader()

        for e in tqdm(range(previous_epoch + 1, self.max_epoch), desc='epoch'):
            for i, inputs in enumerate(tqdm(loader, total=len(loader))):
                obj_val = self.forward(previous_state, i, inputs)
                self.backward(previous_state, obj_val)

            previous_state = TrainerState(
                epoch=e,
                model=previous_state.model,
                optimizer=previous_state.optimizer,
                scheduler=previous_state.scheduler
            )

            self.finalize_epoch(previous_state)
