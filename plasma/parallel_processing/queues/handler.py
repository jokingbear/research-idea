import plasma.functional as F


class ExceptionHandler[T](F.AutoPipe):
    
    def run(self, data:T, e:Exception):
        raise e
