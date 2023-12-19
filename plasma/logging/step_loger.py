from .time_logger import Timer


class StepLogger:

    def __init__(self):
        self.logs = []

    def log_function(self, func, *args, custom_name=None, **kwargs):
        custom_name = custom_name or func.__qualname__

        with Timer(verbose=False) as timer:
            results = func(*args, **kwargs)
            self.logs.append({custom_name: {'results': results, 'time': timer.duration}})

        return results
