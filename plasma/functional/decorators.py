from functools import wraps


def propagate_none(func):

    @wraps(func)
    def alt_func(inputs, **kwargs):
        if inputs is not None:
            return func(inputs, **kwargs)
    
    return alt_func
