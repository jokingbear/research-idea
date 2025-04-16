from queue import Queue
from .signals import Signal


def internal_run(queue:Queue, processor, exception_handler):
    is_not_cancelled = True
    if exception_handler is None:
        exception_handler = _handle_exception

    while is_not_cancelled:
        data = queue.get()
        exception = None

        is_not_cancelled = data is not Signal.CANCEL
        if is_not_cancelled:
            try:
                processor(data)
            except Exception as e:
                exception_handler(e)

        queue.task_done()
        if exception is not None:
            raise exception


def _handle_exception(data, ex:Exception):
    raise ex
