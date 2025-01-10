import multiprocessing as mp

from ...functional import State
from multiprocessing.managers import SyncManager


class Proxy(State):

    def __init__(self, obj, manager:SyncManager=None):
        super().__init__()

        self._manager = manager
        self.obj = obj

        outputs = {} if manager is None else manager.dict()
        self._outputs = outputs
    
    def run(self, process_func, out_key=None):
        try:
            result = process_func(self.obj)
        except Exception as e:
            result = e
            out_key = f'{process_func.__qualname__}_exception'
    
        if out_key is not None:
            self._outputs[out_key] = result
    
    @property
    def results(self):
        return dict(self._outputs)
    
    def release(self):
        del self._outputs
        self._outputs = {} if self._manager is None else self._manager.dict()
