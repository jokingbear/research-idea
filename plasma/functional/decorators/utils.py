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

