import multiprocessing as mp


class ProcessCommunicator:

    def __init__(self, comfunc, nprocess, nitem=0, manager: mp.Manager = None):
        """
        Args:
            comfunc: function to communicate between processes, the last argument must be a process index and queue
            nprocess: number of process to initiate
            nitem: maximum number of item in queue
            manager: context manager
        """
        manager = manager if manager is not None else mp
        self.manager = manager

        queue = manager.Queue(nitem)
        self.queue = queue
        self.processes = [manager.Process(target=comfunc, args=(i, queue)) for i in range(nprocess)]

    def __enter__(self):
        [p.start() for p in self.processes]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        [p.terminate() for p in self.processes]
