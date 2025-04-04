from queue import Queue
from .signals import Signal


def internal_run(queue:Queue, callback):
    is_not_cancelled = True
    while is_not_cancelled:
        data = queue.get()
        exception = None

        is_not_cancelled = data is not Signal.CANCEL
        if is_not_cancelled:
            try:
                callback(data)
            except Exception as e:
                exception = e

        queue.task_done()
        if exception is not None:
            raise exception
