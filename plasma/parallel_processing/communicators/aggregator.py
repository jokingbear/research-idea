import time
import multiprocessing as mp

from ...functional import State
from tqdm.auto import tqdm
from multiprocessing.managers import SyncManager, ValueProxy


class Aggregator(State):

    def __init__(self, total:int, sleep=0.5, manager:SyncManager=None, ignore_none=True, count_none=True):
        super().__init__()
        
        manager = manager or mp.Manager()
        self._results = [] if manager is None else manager.list()
        self._finished:int|ValueProxy[int] = 0 if manager is None else mp.Value('i', 0)

        self._marked_attributes.append('finished')
        self.total = total
        self.sleep = sleep
        self.ignore_none = ignore_none
        self.count_none = count_none
        self._manager = manager
    
    def run(self, data):
        if data is not None or (data is None and self.count_none):
            self.update_step()

        if data is not None or (data is None and not self.ignore_none):
            self._results.append(data)

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
        return list(self._results)

    @property
    def finished(self):
        if isinstance(self._finished, int):
            return self._finished
        
        return self._finished.value

    def release(self):
        self._results = [] if self._manager is None else self._manager.list()
        self._finished = 0 if self._manager is None else mp.Value('i', 0)

    def update_step(self):
        if isinstance(self._finished, int):
            self._finished += 1
        else:
            self._finished.value += 1
