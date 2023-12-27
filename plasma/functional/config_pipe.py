from .pipe import Pipe


class ConfigPipe(Pipe):

    def __setattr__(self, key, value):
        if key != '_marked_attributes':
            assert isinstance(value, (str, int, float, list, tuple, dict)), \
                'attribute must be instance of str, int, float, list, tuple, dict'

            if key not in self._marked_attributes:
                self._marked_attributes.append(key)

        super().__setattr__(key, value)

    def run(self, inputs):
        raise TypeError('config pipe does not have a run method')
