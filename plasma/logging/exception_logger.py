from functools import wraps
from dataclasses import dataclass
from warnings import warn


@dataclass
class ExceptionIO:
    name:str
    args:list
    kwargs:dict
    exception:Exception


class ExceptionLogger:

    IO = ExceptionIO
    
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
                exio = ExceptionIO(name, args, kwargs, e)
                self.log_func(exio)
                
                if self.raise_on_exception:
                    raise e
                
                if callable(self.on_exception_value):
                    return self.on_exception_value(exio)
                return self.on_exception_value

        return run

    @classmethod
    def _deprecated(*_):
        warn('this property is deprecated, please use IO instead')
        return ExceptionIO

    ExceptionIO = property(fget=_deprecated)
