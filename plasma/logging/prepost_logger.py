from functools import wraps
from collections import namedtuple


FuncIO = namedtuple('FuncIO', ['name', 'args', 'kwargs', 'outputs'])


class PrePostLogger:
    IO = FuncIO

    def __init__(self, name=None, pre_func_logger=None, post_func_logger=print) -> None:
        self.name = name
        self.pre_logger = pre_func_logger
        self.post_logger = post_func_logger
    
    def __call__(self, function):
        name = self.name or function.__qualname__
        
        @wraps(function)
        def run(*args, **kwargs):
            io = FuncIO(name, args, kwargs, None)
            self.pre_logger(io)

            results = function(*args, **kwargs)

            io = PrePostLogger.FuncIO(name, args, kwargs, results)
            self.post_logger(io)
            return results

        return run
