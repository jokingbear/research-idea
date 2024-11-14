import threading

from queue import Queue
from .signals import Signal
from .utils import internal_run
from .prototype import QueuePrototype


class ThreadQueue(QueuePrototype[list[threading.Thread]]):

    def __init__(self, persistent=False, n=1):
        super().__init__(block=False)

        self.persistent = persistent
        self.n = n
        self._queue = Queue()

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
        for _ in self._state:
            self.put(Signal.CANCEL)
        self._queue.join()

        for t in self._state:
            t.join()
        super().release()
