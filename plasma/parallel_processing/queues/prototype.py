from ...functional import AutoPipe, chain
from abc import abstractmethod


class QueuePrototype(AutoPipe):

    def __init__(self, block):
        super().__init__()

        self._block = block
        self.__init_private_state()

    def run(self):
        if self._callback is None:
            raise AttributeError('register callback hasn\'t been called on this queue')

        if not self._block and self._state is not None:
            raise AttributeError('queue is already running')

        self._running = True
        self._state = self._init_state()
        
    @abstractmethod
    def _init_state(self):
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

    def __init_private_state(self):
        self._callback = None
        self._state = None
        self._running = False

    def release(self):
        self.__init_private_state()
