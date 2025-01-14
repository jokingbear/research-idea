from queue import Queue
from .signals import Signal


def internal_run(queue:Queue, persistent, callback):
    is_continued = True
    while is_continued:
        data = queue.get()
        exception = None

        is_continued = data is not Signal.CANCEL
        if is_continued:
            try:
                callback(data)
            except Exception as e:
                if persistent:
                    queue.put(data)
                exception = e

        queue.task_done()
        if exception is not None:
            raise exception
