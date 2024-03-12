import torch.multiprocessing as mp


class TorchCommunicator:

    def __init__(self, inference_func, pool_size=2, qsize=0, manager=None, auto_loop=False, start_method='forkserver'):
        """
        Args:
            inference_func: function to run torch, signature: process_id, queue -> ()
            pool_size: number of torch process
            qsize: maximum number of item in queue
            auto_loop: whether to auto run the function in loop
            spawn_method: torch process spawn method, default forkserver, on Windows use spawn
        """
        mp.set_start_method(start_method)
        manager = manager or mp.Manager()
        q = manager.Queue(qsize)
        inference_func = _loop_(inference_func) if auto_loop else inference_func
        tasks = [mp.Process(target=inference_func, args=(i, q)) for i in range(pool_size)]

        self._manager = manager
        self.queue = q
        self.tasks = tasks

    def __enter__(self):
        self._manager.__enter__()
        [t.start() for t in self.tasks]
        return self

    def __exit__(self, *_):
        [t.kill() for t in self.tasks]
        self._manager.__exit__(*_)


class _loop_:

    def __init__(self, running_func) -> None:
        self.running_func = running_func
    
    def __call__(self, i, queue):
        while True:
            self.running_func(i, queue)
