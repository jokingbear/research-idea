from .pipe import Pipe


class AutoPipe(Pipe):

    def __setattr__(self, key, value):
        if key != '_marked_attributes':
            if key not in self._marked_attributes and '_' != key[0]:
                self._marked_attributes.append(key)

        super().__setattr__(key, value)
