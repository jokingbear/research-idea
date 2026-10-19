from apscheduler.schedulers.background import BackgroundScheduler
from collections.abc import Iterable
from typing import Protocol
from threading import Lock

from ..data_model.collections import ZippedStream, Stream


class SSTable[K, V]:
    
    def __init__(self, 
            storage:Storage[K, V], *, 
            sync_interval_seconds:float
        ):
        self.storage = storage
        self._data = dict[K, V]()
        self._deleted = set()
        
        self._lock = Lock()
        scheduler = BackgroundScheduler()
        scheduler.add_job(self._sync, 'interval', seconds=sync_interval_seconds)
        scheduler.start()
        self._scheduler = scheduler
    
    def get(self, k:K):
        if k in self._deleted:
            return 

        data = self._data
        if k not in data:
            (_, value), = self.storage.get([k])
        else:
            value = data[k]
        
        return value
        
    def multi_get(self, *keys:K):
        data = self._data
        inmem_values = {k: data[k] for k in keys if k in data}
        stored_values = dict(self.storage.get(k for k in keys if k not in data))
        
        return (
            Stream(keys)
            .split(lambda k: (k, inmem_values.get(k, stored_values.get(k, None))))
        )

    def set(self, k:K, v:V):
        with self._lock:
            self._data[k] = v
            self._deleted.discard(k)
    
    def delete(self, key:K):
        with self._lock:
            self._data.pop(key, None)
            self._deleted.add(key)
    
    def _sync(self):
        with self._lock:
            data = self._data
            self._data = {}
            
            deleted = self._deleted
            self._deleted = set()

        if len(data) > 0:
            self.storage.put(data.items())
        
        if len(deleted) > 0:
            self.storage.delete(deleted)


class Storage[K, V](Protocol):
    
    def put(self, data:Iterable[tuple[K, V]]) -> None:...
    
    def delete(self, data:Iterable[K]) -> None:...
    
    def get(self, data:Iterable[K]) -> ZippedStream[K, V|None]:...
