from queue import Queue
from .signals import Signal
from .handler import ExceptionHandler


def internal_run(queue:Queue, processor, exception_handler):
    is_not_cancelled = True
    exception_handler = exception_handler or ExceptionHandler()

    while is_not_cancelled:
        data = queue.get()
        exception = None

        is_not_cancelled = data is not Signal.CANCEL
        if is_not_cancelled:
            try:
                processor(data)
            except Exception as e:
                exception_handler(data, e)

        queue.task_done()
        if exception is not None:
            raise exception
