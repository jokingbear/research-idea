from queue import Queue
from .signals import Signal
from .handler import ExceptionHandler


def internal_run(queue:Queue, processor, exception_handler):
    is_not_cancelled = True
    exception_handler = exception_handler or ExceptionHandler()

    while is_not_cancelled:
        data = queue.get()

        is_not_cancelled = data is not Signal.CANCEL
        try:
            if is_not_cancelled:
                processor(data)
        except Exception as e:
            exception_handler(data, e)
        finally:
            queue.task_done()
