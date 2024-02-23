import multiprocessing as mp


class ProcessCommunicator:

    def __init__(self, comfuncs, qsize=0, manager: mp.Manager = None):
        """
        Args:
            comfuncs: list of functions to communicate between processes, the last argument must be a process index and queue
            qsize: maximum number of item in queue
            manager: context manager
        """
        manager = manager if manager is not None else mp
        self.manager = manager

        queue = manager.Queue(qsize)
        self.queue = queue
        self.processes = [manager.Process(target=f, args=(i, queue)) for i, f in enumerate(comfuncs)]

    def __enter__(self):
        [p.start() for p in self.processes]
        return self

    def __exit__(self, *_):
        [p.terminate() for p in self.processes]
