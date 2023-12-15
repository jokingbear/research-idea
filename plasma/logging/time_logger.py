import time

from ..functional import partials


class Timer:

    def __init__(self, verbose=True):
        self.verbose = verbose

        self.start = None

    def __enter__(self):
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print(f'running time: {self.duration:.2f}')
        
        self.start = None

    @property
    def duration(self) -> float:
        if self.start is not None:
            return time.time() - self.start

        raise NotImplementedError('__enter__ method has not been initiated')

    def __call__(self, func):
        return partials(self._proxy_, func)

    def _proxy_(self, func, *args, **kwargs):
        with self.__enter__() as timer:
            results = func(*args, **kwargs)
            print(f'running time of {func} is: {timer.duration:.2f}')
        return results


class StepLogger:

    def __init__(self):
        self.logs = []

    def log_function(self, func, *args, custom_name=None, **kwargs):
        custom_name = custom_name or func.__qualname__

        with Timer(verbose=False) as timer:
            results = func(*args, **kwargs)
            self.logs.append({custom_name: {'results': results, 'time': timer.duration}})

        return results
