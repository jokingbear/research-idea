import torch.multiprocessing as mp

from .base_block import Block


class TorchBlock(Block):

    def __init__(self, inference_funcs, in_queue: mp.JoinableQueue, out_queue:  mp.JoinableQueue):
        """
        Args:
            inference_func: list of function to run torch mp, signature: process_id, in_queue, out_queue -> ()
            in_queue: input queue
            out_queue: output queue
        """
        super().__init__(in_queue, out_queue)
        tasks = [mp.Process(target=f, args=(i, in_queue, out_queue)) for i, f in enumerate(inference_funcs)]
        self.tasks = tasks

    def init(self):
        [t.start() for t in self.tasks]

    def terminate(self, exc_type, exc_val, exc_tb):
        [t.kill() for t in self.tasks]

    @staticmethod
    def set_start_method(method='forkserver'):
        mp.set_start_method(method)
