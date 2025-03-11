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
        def alt_func(inputs):
            if inputs is self.value:
                return self.value
            else:
                return func(inputs)

        return alt_func
