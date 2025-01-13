from queue import Queue
from .signals import Signal


def internal_run(queue:Queue, persistent, callback):
    while True:
        data = queue.get()
        
        if data is Signal.CANCEL:
            break
        else:
            exception = None
            try:
                callback(data)
            except Exception as e:
                if persistent:
                    queue.put(data)
                exception = e

            queue.task_done()
            if exception is not None:
                raise exception
