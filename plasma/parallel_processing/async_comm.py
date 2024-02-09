import asyncio as aio


class TaskCommunicator:

    def __init__(self, comfuncs, nitem=0):
        """
        Args:
            comfunc: function to communicate between processes, the last argument must be a task index and queue
            qsize: maximum number of item in queue
        """

        queue = aio.Queue(nitem)
        self.queue = queue
        self.coroutines = [f(i, queue) for i, f in enumerate(comfuncs)]
        self.tasks = []

    def __enter__(self):
        self.tasks = [aio.Task(c) for c in self.coroutines]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        [t.cancel() for t in self.tasks]
