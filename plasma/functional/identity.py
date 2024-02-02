from .pipe import Pipe


class Identity(Pipe):

    def run(self, *inputs):
        return inputs