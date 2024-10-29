import logging

from queue import Queue
from .signals import Signal


def internal_run(queue:Queue, persistent, callback):
    while True:
        data = queue.get()

        if data != Signal.CANCEL:
            try:
                callback(data)
            except Exception as e:
                logging.error(e)
                if persistent:
                    queue.put(data)

        queue.task_done()

        if data == Signal.CANCEL:
            break
