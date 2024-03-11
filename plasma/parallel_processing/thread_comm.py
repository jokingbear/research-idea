import concurrent.futures as cf

from os import cpu_count
from queue import Queue


class ThreadCommunicator:

    def __init__(self, comfuncs, pool_size=None, qsize=0):
        """
        Args:
            comfuncs: list of functions to communicate between processes, the last argument must be a process index and queue
            pool_size: size of pool thread, default = comfuncs
            qsize: maximum number of item in queue
        """
        max_cpu = cpu_count() or 0
        max_cpu = max(32, max_cpu + 4)
        pool_size = pool_size or len(comfuncs)
        pool = cf.ThreadPoolExecutor(max_workers=min(pool_size, max_cpu))
        q = Queue(qsize)
        tasks = [pool.submit(_loop_(f), i, q) for i, f in enumerate(comfuncs)]

        self.pool = pool
        self.queue = q
        self.tasks = tasks

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, *_):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pool.__exit__(*_)


class _loop_:

    def __init__(self, running_func) -> None:
        self.running_func = running_func
    
    def __call__(self, i, queue: Queue):
        while True:
            self.running_func(i, queue)
