from .pipe import Pipe


class SequentialPipe(Pipe):

    def __setattr__(self, key, value):
        if key != '_marked_attributes':
            assert isinstance(value, Pipe), 'attribute must be instance of Pipe'

            if key not in self._marked_attributes:
                self._marked_attributes.append(key)

        super().__setattr__(key, value)

    def run(self, inputs):
        for attr in self._marked_attributes:
            p = getattr(self, attr)
            inputs = p(inputs)

        return inputs
