import concurrent.futures as cf
import multiprocessing as mp
import threading

from queue import Queue
from .base_block import Block


class CPUBlock(Block):

    def __init__(self, comfuncs, input_queue: Queue, output_queue:Queue, use_thread=False):
        """
        Args:
            comfuncs: list of functions to communicate between processes, the last argument must be a process index and queue
            input_queue: queue for input
            output_queue: queue for output
            use_thread: whether to run comfuncs on different thread or process
            auto_loop: whether to run comfuncs in infinite loop
        """       
        super().__init__(input_queue, output_queue)
        if use_thread:
            factory = threading.Thread
        else:
            factory = mp.Process

        tasks = [factory(target=f, args=(i, input_queue, output_queue)) for i, f in enumerate(comfuncs)]
        self.tasks = tasks

    def init(self):
        [t.start() for t in self.tasks]

    def terminate(self, *_):
        for t in self.tasks:
            if isinstance(t, mp.Process):
                t.kill()
