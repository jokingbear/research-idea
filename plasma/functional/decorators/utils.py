import inspect

from functools import wraps


def automap(func):
    signature = inspect.getfullargspec(func)
    is_instance_method = signature.args[0] == 'self'

    @wraps(func)
    def instance_alt_func(self, inputs):
        if isinstance(inputs, (tuple, list)):
            return func(self, *inputs)
        elif isinstance(inputs, dict):
            return func(self, **inputs)
        elif inputs is None:
            return func(self)
        else:
            return func(self, inputs)

    @wraps(func)
    def alt_func(inputs):
        if isinstance(inputs, (tuple, list)):
            return func(*inputs)
        elif isinstance(inputs, dict):
            return func(**inputs)
        elif inputs is None:
            return func()
        else:
            return func(inputs)

    if is_instance_method:
        return instance_alt_func
    else:
        return alt_func


class propagate:

    def __init__(self, value=None):
        self.value = value
    
    def __call__(self, func):
        signature = inspect.getfullargspec(func)
        is_instance_method = signature.args[0] == 'self'
        offset = 1 if is_instance_method else 0
        args = signature.args[offset:]
        defaults = len(signature.defaults or [])
        total_mandatory = len(args) - defaults

        assert total_mandatory == 1, \
            f'this decorator only applies to instance one arg or one arg method, current signature {args}'

        propagator = self
        @wraps(func)    
        def alt_func(inputs):
            if inputs is propagator.value:
                return propagator.value
            else:
                return func(inputs)
        
        @wraps(func)    
        def instance_alt_func(self, inputs):
            if inputs is propagator.value:
                return None
            else:
                return func(self, inputs)

        return instance_alt_func if is_instance_method else alt_func
