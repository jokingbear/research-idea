import multiprocessing as mp

from ...functional import State
from multiprocessing.managers import SyncManager


class ObjectInquirer(State):

    def __init__(self, obj, manager:SyncManager):
        super().__init__()

        self._manager = manager
        self.obj = obj

        outputs = manager.dict()
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
        self._outputs = self._manager.dict()

    def __getattribute__(self, name):
        if name[0] != '_' and name not in {'run', 'release', 'results', '__init__'} and hasattr(self.obj, name):
            return self.run()

        return super().__getattribute__(name)
