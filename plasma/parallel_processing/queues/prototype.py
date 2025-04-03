from .base import Queue
from warnings import warn


class QueuePrototype[T](Queue[T]):

    def __init__(self, block):
        super().__init__(block=block)

        warn('this class is deprecated, please use Queue')
