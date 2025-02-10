from .pipe import AutoPipe
from abc import abstractmethod


class SimplePipe[I, O](AutoPipe):

    @abstractmethod
    def run(self, inputs:I)->O:
        pass

    def __call__(self, inputs:I) -> O:
        return self.run(inputs)
