import time

from functools import wraps
from collections import namedtuple


class Timer:
    TimeIO = namedtuple('TimeIO', ['name', 'duration', 'args', 'kwargs'])

    def __init__(self, log_func=print, log_inputs=False) -> None:
        self.log_func = log_func
        self.log_inputs = log_inputs
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self._end = time.time()
        if self.log_func is None:
            print(f'{self.duration:.2f}s')

    @property
    def duration(self):
        if self._start is None or self._end is None:
            raise ValueError('please call enter and exit method appropriately')
        return self._end - self._start

    def __call__(self, func):
        name = func.__qualname__
        
        @wraps(func)
        def run_timer(*args, **kwargs):
            with self:
                results = func(*args, **kwargs)
            timeio = self.TimeIO(name, self.duration, args if self.log_inputs else [], kwargs if self.log_inputs else {})
            self.log_func(timeio)
            return results

        return run_timer
