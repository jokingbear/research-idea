import inspect

from functools import wraps


def automap(func):
    signature = inspect.getfullargspec(func)
    is_instance_method = signature.args[0] == 'self'

    def alt_func(*inputs):
        instance = []
        if is_instance_method:
            instance = [inputs[0]]
            inputs = inputs[1]

        if isinstance(inputs, (tuple, list)):
            return func(*instance, *inputs)
        elif isinstance(inputs, dict):
            return func(*instance, **inputs)
        elif inputs is None:
            return func(*instance)
        else:
            return func(*instance, inputs)

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
