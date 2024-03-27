from .functional import parallel_iterate
from .cpu_block import CPUBlock
from .torch_block import TorchBlock
from . import utils

from queue import Queue as ThreadQueue
from multiprocessing import JoinableQueue, SimpleQueue
from torch.multiprocessing import JoinableQueue as TorchJoinableQueue, SimpleQueue as TorchSimpleQueue
