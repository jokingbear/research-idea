from .pipe import AutoPipe


class TuplePipe(AutoPipe):
    
    def __iter__(self):
        for a in self._marked_attributes:
            yield getattr(self, a)
