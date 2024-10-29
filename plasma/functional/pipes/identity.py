from .pipe import AutoPipe


class Identity(AutoPipe):

    def run(self, *inputs):
        if len(inputs) == 0:
            return None
        elif len(inputs) == 1:
            return inputs[0]
        else:
            return inputs
