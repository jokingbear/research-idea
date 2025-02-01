from .pipe import AutoPipe
from abc import abstractmethod


class State(AutoPipe):

    @abstractmethod
    def release(self):
        pass
