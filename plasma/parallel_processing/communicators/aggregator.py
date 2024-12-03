import time

from ...functional import State
from tqdm.auto import tqdm


class Aggregator(State):

    def __init__(self, total:int, sleep=0.5, manager=None):
        super().__init__()

        self._results = [] if manager is None else manager.list()
        self._marked_attributes.append('finished')
        self.total = total
        self.sleep = sleep
        self._manager = manager
    
    def run(self, data):
        self._results.append(data)
    
    @property
    def finished(self):
        return len(self._results)

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
        return [*self._results]

    def release(self):
        self._results = [] if self._manager is None else self._manager.list()
