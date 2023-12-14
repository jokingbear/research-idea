import time
import numpy as np


class Time:

    def __init__(self, verbose=True):
        self.verbose = verbose

        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print(self.duration)
        
        self.start = None
        self.end = None
    
    @property
    def duration(self) -> float:
        if self.start is not None:
            return time.time() - self.start

        return np.nan


class StepLogger:

    def __init__(self):
        self.logs = []

    def log_function(self, func, *args, custom_name=None, **kwargs):
        custom_name = custom_name or func.__qualname__

        with Time(verbose=False) as timer:
            results = func(*args, **kwargs)
            self.logs.append({custom_name: {'results': results, 'time': timer.duration}})

        return results