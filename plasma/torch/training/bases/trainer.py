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
    rank = 0
    world_size = 1
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
    def backward(self, i:int, inputs, objective_val:torch.Tensor):
        pass

    def optimize(self, i:int, inputs, objective_val:torch.Tensor):
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
        for e in tqdm(range(current_epoch, self.max_epoch), desc='epoch', disable=self.rank != 0):
            self.current_epoch = e
            for i, inputs in enumerate(tqdm(loader, total=len(loader))):
                obj_val = self.forward(i, inputs)
                self.backward(i, inputs, obj_val)
                self.optimize(i, inputs, obj_val)

            self.finalize_epoch()
    
    def load_state(self, state:dict):
        self.current_epoch = state['current_epoch']
        self.current_iteration = state['current_iteration']
        self.max_epoch = state['max_epoch']
        print(self.model.load_state_dict(state['model_state']))
        print(self.optimizer.load_state_dict(state['optimizer_state']))

        if self.scheduler is not None:
            print(self.scheduler.load_state_dict(state['scheduler_state']))
