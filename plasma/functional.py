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

    def __init__(self, func, *args, first=True, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.first = first

    def __call__(self, *new_args, **new_kwargs):
        if self.first:
            return self.func(*self.args, *new_args, **self.kwargs, **new_kwargs)
        else:
            return self.func(*new_args, *self.args, **new_kwargs, **self.kwargs)
