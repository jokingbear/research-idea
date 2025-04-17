import multiprocessing as mp

from .signals import Signal
from .utils import internal_run
from .base import Queue
from ...functional.decorators import propagate


class ProcessQueue(Queue[list[mp.Process]]):

    def __init__(self, n=1, name=None, qsize=0, timeout=None):
        super().__init__(name, n)

        self._queue = mp.JoinableQueue(qsize)
        self.timeout = timeout

    def _init_state(self):
        processes = [mp.Process(target=internal_run, args=(self._queue, self._callback, self._exception_handler)) 
                     for _ in range(self.num_runner)]
        [p.start() for p in processes]
        return processes

    @propagate(Signal.IGNORE)
    def put(self, x):
        self._queue.put(x, block=True, timeout=self.timeout)
    
    def release(self):
        self._queue.join()
        if self._state is not None:
            for _ in self._state:
                self.put(Signal.CANCEL)
            self._queue.join()

            for p in self._state:
                p.join()
                p.terminate()
        
        super().release()
