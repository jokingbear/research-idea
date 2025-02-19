import time
import multiprocessing as mp
import threading

from ...functional import State
from tqdm.auto import tqdm
from multiprocessing.managers import SyncManager, ValueProxy
from multiprocessing import shared_memory
from ..queues import Signal


class Aggregator(State):

    def __init__(self, total:int, sleep=0.5, manager:SyncManager=None, ignore_none=True, count_none=True):
        super().__init__()

        self._results = []
        
        process_queue = None if manager is None else mp.JoinableQueue()
        self._process_queue = process_queue

        self._finished:int|ValueProxy[int] = 0 if manager is None else mp.Value('i', 0)
  
        self._marked_attributes.append('finished')
        self.total = total
        self.sleep = sleep
        self.ignore_none = ignore_none
        self.count_none = count_none
        self._manager = manager
    
    def run(self, data):
        if data is not None or (data is None and self.count_none):
            self._update_step()

        if data is not None or (data is None and not self.ignore_none):
            if self._process_queue is not None:
                self._process_queue.put(data)
            else:
                self._aggregate(data)

    def wait(self, **tqdm_kwargs):
        with tqdm(total=self.total, **tqdm_kwargs) as prog:
            n = self.finished
            prog.update(n)
            while self.finished != self.total:
                time.sleep(self.sleep)
                new_n = self.finished
                diff = new_n - n
                n = new_n
                prog.update(diff)

    @property
    def results(self):
        if self._process_queue is not None:
            try:
                while True:
                    self._aggregate(self._process_queue.get(timeout=0.01))
            except:
                pass

        return self._results.copy()

    @property
    def finished(self):
        if isinstance(self._finished, int):
            return self._finished
        
        return self._finished.value

    def release(self):
        self._results = []
        self._finished = 0 if self._manager is None else mp.Value('i', 0)

    def _update_step(self):
        if isinstance(self._finished, int):
            self._finished += 1
        else:
            self._finished.value += 1

    def _aggregate(self, data):
        self._results.append(data)
