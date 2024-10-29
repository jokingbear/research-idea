import time

from ..functional import partials, proxy_func
from functools import wraps


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
        if self.log_func is None:
            print(self.duration)

    @property
    def duration(self):
        if self.start is None or self.end is None:
            raise ValueError('please call enter and exit method appropriately')
        return self.end - self.start

    def __call__(self, func):
        name = func.__qualname__
        
        @wraps(func)
        def run_timer(*args, **kwargs):
            with self:
                results = func(*args, **kwargs)
            duration = self.duration
            self.log_func({name: f'{duration:.2f}s'})
            return results

        return run_timer
