from .pipe import AutoPipe


class LambdaPipe(AutoPipe):
    
    def __init__(self, func):
        super().__init__()

        self.func = func
    
    def run(self, *inputs, **kwargs):
        return self.func(*inputs, **kwargs)
