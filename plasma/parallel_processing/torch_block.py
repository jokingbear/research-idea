import torch.multiprocessing as mp
import time

from .base_block import Block


class TorchBlock(Block):

    def __init__(self, inference_funcs, in_queue: mp.JoinableQueue, out_queue:  mp.JoinableQueue):
        """
        Args:
            inference_func: list of function to run torch mp, signature: process_id, init_event, in_queue, out_queue -> ()
            in_queue: input queue
            out_queue: output queue
        """
        super().__init__(in_queue, out_queue)

        events = [mp.Event() for i, _ in enumerate(inference_funcs)]
        tasks = [mp.Process(target=f, args=(i, e, in_queue, out_queue)) for i, (e, f) in enumerate(zip(events, inference_funcs))]
        self.tasks = tasks
        self.init_events = events

    def init(self):
        [t.start() for t in self.tasks]
        while not all(e.is_set() for e in self.init_events):
            time.sleep(0.5)


    def terminate(self, exc_type, exc_val, exc_tb):
        [t.kill() for t in self.tasks]

    @staticmethod
    def set_start_method(method='forkserver'):
        mp.set_start_method(method)
