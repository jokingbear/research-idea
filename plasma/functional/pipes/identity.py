from .pipe import AutoPipe


class Identity[T](AutoPipe):

    def run(self, x:T):
        return x
