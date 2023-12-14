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


class partials:

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *new_args, **new_kwargs):
        return self.func(*self.args, *new_args, **self.kwargs, **new_kwargs)
