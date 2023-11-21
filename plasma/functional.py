def auto_map_func(func):

    def auto_input(inputs):
        if isinstance(inputs, (tuple, list)):
            return func(*inputs)
        elif isinstance(inputs, dict):
            return func(**inputs)
        elif inputs is None:
            return func()
        else:
            return func(inputs)
    
    return auto_input


def partials(func, *args, **kwargs):

    def _tmp_func(*new_args, **new_kwargs):
        return func(*new_args, *args, **kwargs, **new_kwargs)

    return _tmp_func


class StepLogger:

    def __init__(self):
        self.logs = []

    def log_function(self, func, *args, custom_name=None, **kwargs):
        custom_name = custom_name or func.__qualname__

        results = func(*args, **kwargs)

        self.logs.append({custom_name: results})

        return results
