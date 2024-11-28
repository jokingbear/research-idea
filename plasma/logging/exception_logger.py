from functools import wraps
from collections import namedtuple


class ExceptionLogger:
    ExceptionIO = namedtuple('ExIO', ['name', 'args', 'kwargs', 'exception'])

    def __init__(self, name=None, log_func=print, raise_on_exception=True, on_exception_return=None) -> None:
        self.name = name
        self.log_func = log_func
        self.raise_on_exception = raise_on_exception
        self.on_exception_value = on_exception_return
    
    def __call__(self, function):
        name = self.name or function.__qualname__
        
        @wraps(function)
        def run(*args, **kwargs):
            try:
                results = function(*args, **kwargs)
                return results
            except Exception as e:
                exio = self.ExceptionIO(name, args, kwargs, e)
                self.log_func(exio)
                
                if self.raise_on_exception:
                    raise e
                
                if callable(self.on_exception_value):
                    return self.on_exception_value(exio)
                return self.on_exception_value

        return run
