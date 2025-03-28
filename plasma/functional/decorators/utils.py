import inspect

from functools import wraps


def automap(func):
    signature = inspect.signature(func)
    is_instance_method = 'self' in signature.parameters

    @wraps(func)
    def alt_func(*args, **kwargs):
        if is_instance_method:
            instance = args[:1]
            inputs = args[1]
        else:
            inputs = args[0]
            instance = []

        if isinstance(inputs, (tuple, list)):
            return func(*instance, *inputs, **kwargs)
        elif isinstance(inputs, dict):
            return func(**inputs, **kwargs)
        elif inputs is None:
            return func(*instance)
        else:
            return func(*instance, inputs)

    return alt_func


class propagate:

    def __init__(self, value=None):
        self.value = value
    
    def __call__(self, func):
        signature = inspect.signature(func)

        is_instance_method = 'self' in signature.parameters
        propagator = self

        @wraps(func)    
        def alt_func(*args, **kwargs):
            if is_instance_method:
                inputs = args[1]
            else:
                inputs = args[0]

            if inputs is propagator.value:
                return propagator.value
            else:
                return func(*args, **kwargs)

        return alt_func
