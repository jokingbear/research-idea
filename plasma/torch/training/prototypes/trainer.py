import torch
import torch.utils
import torch.nn as nn
import torch.optim as opts

from ....functional import AutoPipe
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from abc import abstractmethod


class BaseTrainer(AutoPipe):
    current_epoch = None
    max_epoch:int
    _model:nn.Module
    _optimizer:opts.optimizer.Optimizer
    scheduler:opts.lr_scheduler.LRScheduler

    @abstractmethod
    def init_train_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def forward(self, i:int, inputs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def backward(self, bjective_val:torch.Tensor):
        pass
    
    def finalize_iteration(self):
        pass
    
    def finalize_epoch(self):
        pass

    def run(self):
        loader = self.init_train_loader()
        
        if self.current_epoch is None:
            current_epoch = 0
        else:
            current_epoch = self.current_epoch + 1

        for e in tqdm(range(current_epoch, self.max_epoch), desc='epoch'):
            self.current_epoch = e
            for i, inputs in enumerate(tqdm(loader, total=len(loader))):
                obj_val = self.forward(i, inputs)
                self.backward(obj_val)
                self.finalize_iteration()

            self.finalize_epoch()

    @property
    def state(self):
        return {
            'current_epoch': self.current_epoch,
            'current_iteration': self.current_iteration,
            'max_epoch': self.max_epoch,
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
        }
    
    def load_state(self, state:dict):
        self.current_epoch = state['current_epoch']
        self.current_iteration = state['current_iteration']
        self.max_epoch = state['max_epoch']
        print(self._model.load_state_dict(state['model_state']))
        print(self._optimizer.load_state_dict(state['optimizer_state']))

        if self.scheduler is not None:
            print(self.scheduler.load_state_dict(state['scheduler_state']))
