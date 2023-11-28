import time


class auto_map_func:

    def __init__(self, func):
        self.func = func

    def __call__(self, inputs):
        if isinstance(inputs, (tuple, list)):
            return self.func(*inputs)
        elif isinstance(inputs, dict):
            return self.func(**inputs)
        elif inputs is None:
            return self.func()
        else:
            return self.func(inputs)


def partials(func, *args, **kwargs):

    def _tmp_func(*new_args, **new_kwargs):
        return func(*new_args, *args, **kwargs, **new_kwargs)

    return _tmp_func


class StepLogger:

    def __init__(self):
        self.logs = []

    def log_function(self, func, *args, custom_name=None, **kwargs):
        custom_name = custom_name or func.__qualname__

        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()

        self.logs.append({custom_name: {'results': results, 'time': end - start}})

        return results
