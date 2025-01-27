from .base import Queue


class QueuePrototype[T](Queue[T]):

    def __init__(self, block):
        super().__init__()

        print('this class is deprecated, please use Queue')
