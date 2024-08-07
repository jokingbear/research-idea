import time

from ..functional import partials, proxy_func


class Timer:

    def __init__(self, log_func=print) -> None:
        self.log_func = log_func
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        self.end = time.time()

    @property
    def duration(self):
        if self.start is None or self.end is None:
            raise ValueError('please call enter and exit method appropriately')
        return self.end - self.start

    def __call__(self, func):
        return _timer_proxy(self, func)


class _timer_proxy(proxy_func):

    def __init__(self, timer: Timer, func):
        super().__init__(func)
        self.timer = timer
        self.name = func.__qualname__
    
    def __call__(self, *args, **kwds):
        with self.timer as timer:
            results = self.func(*args, **kwds)
        duration = timer.duration
        self.timer.log_func({self.name: f'{duration:.2f}s'})
        return results
    
    def __get__(self, instance, owner):
        return partials(self, instance)
