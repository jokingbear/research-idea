from ...functional import State, chain
from abc import abstractmethod
from warnings import warn


class Queue[T](State):

    def __init__(self, name=None, n=1, block=None):
        super().__init__()

        if block is not None:
            warn('block is deprecated')
            self._block = block

        self.name = name
        self.num_runner = n
        self._running = False
        self.__clean_state()

    def run(self):
        if self._callback is None:
            raise AttributeError('register_callback has not been called on this queue')
        
        if not self._running:
            self._running = True
            self._state = self._init_state()
        return self
        
    @abstractmethod
    def _init_state(self) -> T:
        pass

    @abstractmethod
    def put(self, x):
        pass

    def register_callback(self, callback):
        assert not self._running,\
            'queue is already running, please release it to register new function'
        self._callback = callback
        if self.name is None:
            if hasattr(callback, '__name__'):
                self.name = callback.__name__
            else:
                self.name = type(callback).__name__

        return self

    def chain(self, callback):
        assert not self._running, \
            'queue is already running, please release it to chain new function'
        self._callback = chain(self._callback, callback)
        return self

    def release(self):
        self.__clean_state()

    def __clean_state(self):
        self._state = None
        self._running = False

    @property
    def running(self):
        return self._running
