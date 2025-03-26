import time
import datetime

from functools import wraps
from dataclasses import dataclass


@dataclass
class Counter:
    start: datetime.datetime
    end: datetime.datetime
    duration: datetime.timedelta


@dataclass
class TimeIO:
    name:str
    timer: Counter
    args:list
    kwargs:dict


class Timer:

    IO = TimeIO

    def __init__(self, log_func=print, log_inputs=False) -> None:
        self.log_func = log_func
        self.log_inputs = log_inputs
        self._start = None
        self._end = None
        self.io = TimeIO

    def __enter__(self):
        self._end = None
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
    
        return datetime.timedelta(seconds=self._end - self._start)

    @property
    def start(self):
        return datetime.datetime.fromtimestamp(self._start)
    
    @property
    def end(self):
        if self._end is None:
            return None

        return datetime.datetime.fromtimestamp(self._end)

    def check(self):
        if self._start is None:
            raise ValueError('please call enter and exit method appropriately')
        
        if self._end is not None:
            return self.duration

        return datetime.timedelta(seconds=time.time() - self._start)

    def __call__(self, func):
        name = func.__qualname__
        
        @wraps(func)
        def run_timer(*args, **kwargs):
            with self:
                results = func(*args, **kwargs)
            timeio = TimeIO(name, Counter(self.start, self.end, self.duration), 
                            args if self.log_inputs else [], kwargs if self.log_inputs else {})
            self.log_func(timeio)
            return results

        return run_timer

    def __repr__(self):
        return f'(start={self.start}, end={self.end}, duration={self.duration})'
