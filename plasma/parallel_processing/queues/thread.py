import threading
import queue

from .signals import Signal
from .utils import internal_run
from .base import Queue


class ThreadQueue(Queue[list[threading.Thread]]):

    def __init__(self, qsize=0, n=1, persistent=False):
        super().__init__(block=False)

        self.persistent = persistent
        self.n = n
        self._queue = queue.Queue(qsize)

    def _init_state(self):
        if self._callback is None:
            raise AttributeError('there is no registered callback for this queue.')
        
        threads = [threading.Thread(target=internal_run, args=(self._queue, self.persistent, self._callback)) for i in range(self.n)]
        [t.start() for t in threads]
        return threads

    def put(self, x):
        self._queue.put(x)
    
    def release(self):
        self._queue.join()
        if self._state is not None:
            for _ in self._state:
                self.put(Signal.CANCEL)
            self._queue.join()

            for t in self._state:
                t.join()
        super().release()

    def join(self):
        self._queue.join()
