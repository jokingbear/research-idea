import time
import multiprocessing as mp

from ...functional import State
from tqdm.auto import tqdm
from multiprocessing.managers import SyncManager, ValueProxy
from warnings import warn


class Aggregator(State):

    def __init__(self, total:int, sleep=1e-2, process_base=False, ignore_none=True, count_none=True):
        super().__init__()
        self._results = []

        process_base = process_base
        process_queue = None if not process_base else mp.JoinableQueue()
        self._process_queue = process_queue
        self._finished:int|ValueProxy[int] = 0 if not process_base else mp.Value('i', 0)

        self._marked_attributes.append('finished')
        self.total = total
        self.sleep = sleep
        self.process_base = process_base
        self.ignore_none = ignore_none
        self.count_none = count_none
    
    def run(self, data):
        if data is not None or (data is None and self.count_none):
            self._update_step()

        if data is not None or (data is None and not self.ignore_none):
            self._aggregate(data)

        if self._process_queue is not None and self._finished.value == self.total:
            self._process_queue.put(self._results)
        
        if self.finished == self.total:
            return self._results

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
        
        if self._process_queue is not None:
            self._results = self._process_queue.get()

        return self.results

    @property
    def results(self):
        return self._results.copy()

    @property
    def finished(self):
        if isinstance(self._finished, int):
            return self._finished
        
        return self._finished.value

    def release(self):
        self._results = []
        self._finished = 0 if not self.process_base else mp.Value('i', 0)

    def _update_step(self):
        if isinstance(self._finished, int):
            self._finished += 1
        else:
            self._finished.value += 1

    def _aggregate(self, data):
        self._results.append(data)
