from ...functional import State, chain
from abc import abstractmethod
from warnings import warn
from .handler import ExceptionHandler


class Queue[T](State):

    def __init__(self, name=None, num_runner=1):
        super().__init__()

        self.name = name
        self.num_runner = num_runner
        self._running = False
        self.__clean_state()
        self._callback = None
        self._exception_handler = ExceptionHandler()

    def run(self):
        if self._callback is None:
            raise AttributeError('register_callback has not been called on this queue')

        handler = self._exception_handler or ExceptionHandler()
        self._exception_handler = handler
        
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

        return self

    def chain(self, callback):
        assert not self._running, \
            'queue is already running, please release it to chain new function'
        self._callback = chain(self._callback, callback)
        return self

    def on_exception(self, handler:ExceptionHandler):
        assert not self._running, \
            'queue is already running, please release it to register new exception handler'
        self._exception_handler = handler
        
        return self
    
    def release(self):
        self.__clean_state()

    def __clean_state(self):
        self._state = None
        self._running = False

    @property
    def running(self):
        return self._running

    def is_alive(self):
        return False
