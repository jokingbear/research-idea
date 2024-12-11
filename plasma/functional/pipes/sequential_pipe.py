from .state import State


class SequentialPipe[T](State[T]):

    def __init__(self, **pipes):
        super().__init__()

        for key, pipe in pipes.items():
            self.__setattr__(key, pipe)

    def __setattr__(self, key:str, value):
        if key[0] != '_':
            assert callable(value), 'attribute must be instance of AutoPipe or a function'

        super().__setattr__(key, value)

    def run(self, inputs):
        for attr in self._marked_attributes:
            p = getattr(self, attr)
            inputs = p(inputs)

        return inputs
    
    def reset(self):
        for p in self._marked_attributes:
            if isinstance(p, State):
                p.reset()
