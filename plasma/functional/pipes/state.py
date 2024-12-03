from .pipe import AutoPipe
from abc import abstractmethod


class State[T](AutoPipe[T]):

    @abstractmethod
    def release(self):
        pass
