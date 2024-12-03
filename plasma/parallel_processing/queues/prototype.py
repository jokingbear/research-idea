from ...functional import State, chain
from abc import abstractmethod


class QueuePrototype[T](State):

    def __init__(self, block):
        super().__init__()

        self._block = block
        self.__clean_state()

    def run(self):
        if self._callback is None:
            raise AttributeError('register_callback has not been called on this queue')

        if not self._block and self._state is not None:
            raise AttributeError('queue is already running')

        self._running = True
        self._state = self._init_state()
        
    @abstractmethod
    def _init_state(self) -> T:
        pass

    @abstractmethod
    def put(self, x):
        pass

    def register_callback(self, callback):
        self._callback = callback

    def chain(self, callback):
        if self._running:
            raise RuntimeError('queue is already running, please release it to chain new function')
        self._callback = chain(self._callback, callback)

    def release(self):
        self.__clean_state()

    def __clean_state(self):
        self._state = None
        self._running = False

    @property
    def running(self):
        return self._running
