from functools import wraps
from collections import namedtuple
from warnings import deprecated


@deprecated('this class is deprecated, use PrePostLogger')
class FunctionLogger:
    FuncIO = namedtuple('FuncIO', ['name', 'args', 'kwargs', 'outputs'])

    def __init__(self, name=None, log_func=print) -> None:
        self.name = name
        self.log_func = log_func
    
    def __call__(self, function):
        name = self.name or function.__qualname__
        
        @wraps(function)
        def run(*args, **kwargs):
            results = function(*args, **kwargs)
            funcio = FunctionLogger.FuncIO(name, args, kwargs, results)
            self.log_func(funcio)
            return results

        return run
