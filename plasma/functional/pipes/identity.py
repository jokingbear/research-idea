from .pipe import AutoPipe


class Identity[T](AutoPipe[T]):

    def run(self, x:T):
        return x
