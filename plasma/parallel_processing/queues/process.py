import multiprocessing as mp

from .signals import Signal
from .utils import internal_run
from .base import Queue


class ProcessQueue(Queue[list[mp.Process]]):

    def __init__(self, n=1, persistent=False):
        super().__init__(block=False)

        self.persistent = persistent
        self.n = n
        
        self._queue = mp.JoinableQueue()

    def _init_state(self):
        processes = [mp.Process(target=internal_run, args=(self._queue, self.persistent, self._callback)) for _ in range(self.n)]
        [p.start() for p in processes]
        return processes

    def put(self, x):
        self._queue.put(x)
    
    def release(self):
        self._queue.join()
        if self._state is not None:
            for _ in self._state:
                self.put(Signal.CANCEL)
            self._queue.join()

            for p in self._state:
                p.join()
                p.close()
        
        super().release()

    def join(self):
        self._queue.join()
