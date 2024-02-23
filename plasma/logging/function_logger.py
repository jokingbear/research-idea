import logging

from ..functional import partials, proxy_func


class FunctionLogger:

    def __init__(self, name=None, log_func=logging.info) -> None:
        self.name = name
        self.log_func = log_func
    
    def __call__(self, function):
        return _function_proxy(self, function)
    
    def _process_results(self, results):
        return results


class _function_proxy(proxy_func):

    def __init__(self, function_logger: FunctionLogger, function) -> None:
        super().__init__(function)
        if function_logger.name is None:
            name = function.__qualname__
        self.name = name
        self.logger = function_logger
    
    def __call__(self, *args, **kwargs):
        results = self.func(*args, **kwargs)
        results = self.logger._process_results(results)
        self.logger.log_func({self.name: results})
        return results

    def __get__(self, instance, owner):
        return partials(self, instance)
