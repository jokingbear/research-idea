from functools import wraps


def automap(func):

    def alt_func(inputs):
        if isinstance(inputs, (tuple, list)):
            return func(*inputs)
        elif isinstance(inputs, dict):
            return func(**inputs)
        elif inputs is None:
            return func()
        else:
            return func(inputs)

    return alt_func


class propagate:

    def __init__(self, value=None):
        self.value = value
    
    def __call__(self, func):

        @wraps(func)    
        def alt_func(*inputs):
            if inputs[-1] is self.value:
                return self.value
            else:
                return func(*inputs)

        return alt_func
