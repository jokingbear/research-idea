import logging

from queue import Queue
from .signals import Signal


def internal_run(queue:Queue, persistent, callback):
    while True:
        data = queue.get()

        try:
            if data != Signal.CANCEL:
                callback(data)
        except Exception as e:
            queue.task_done()

            if persistent:
                print(e)
                queue.put(data)
            else: 
                raise e
        finally:
            queue.task_done()

            if data == Signal.CANCEL:
                break
