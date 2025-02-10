import torch
import torch.utils
import torch.nn as nn

from ....functional import AutoPipe
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from abc import abstractmethod
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Trainer(AutoPipe):
    current_epoch = -1
    max_epoch:int
    model:nn.Module
    optimizer:Optimizer
    scheduler:LRScheduler
    scaler: torch.GradScaler = None

    @abstractmethod
    def init_train_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def forward(self, i:int, inputs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def backward(self, objective_val:torch.Tensor):
        pass

    def optimize(self, objective_val:torch.Tensor):
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
    
    def finalize_epoch(self):
        pass

    def run(self):
        loader = self.init_train_loader()
        current_epoch = self.current_epoch + 1
        for e in tqdm(range(current_epoch, self.max_epoch), desc='epoch'):
            self.current_epoch = e
            for i, inputs in enumerate(tqdm(loader, total=len(loader))):
                obj_val = self.forward(i, inputs)
                self.backward(obj_val)
                self.optimize(obj_val)

            self.finalize_epoch()

    @property
    def state(self):
        return {
            'current_epoch': self.current_epoch,
            'current_iteration': self.current_iteration,
            'max_epoch': self.max_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
        }
    
    def load_state(self, state:dict):
        self.current_epoch = state['current_epoch']
        self.current_iteration = state['current_iteration']
        self.max_epoch = state['max_epoch']
        print(self.model.load_state_dict(state['model_state']))
        print(self.optimizer.load_state_dict(state['optimizer_state']))

        if self.scheduler is not None:
            print(self.scheduler.load_state_dict(state['scheduler_state']))
