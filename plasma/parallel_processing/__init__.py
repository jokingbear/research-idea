import multiprocessing as mp

from .functional import parallel_iterate
from .process_comm import ProcessCommunicator
from .async_comm import TaskCommunicator


def create_context():
    return mp.Manager()
