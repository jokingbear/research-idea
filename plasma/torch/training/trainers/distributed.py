import os
import torch.distributed as dist

from ..bases import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedTrainer(Trainer):

    def __init__(self, rank, world_size, 
                 master_addr='localhost', master_port='12355', backend='gloo'):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    def run(self):
        super().run()

        dist.destroy_process_group()
