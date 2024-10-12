from .pipe import AutoPipe
from warnings import warn


class Pipe(AutoPipe):

    def __init__(self, **kwargs):
        super().__init__()

        warn('this class is included for backward compatibility, it will be removed in the future')
        for k, v in kwargs.items():
            self.__setattr__(k, v)
