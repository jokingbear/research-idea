import numpy as np

from typing import Any, cast, overload
from collections.abc import Callable, Iterable, Sequence

from .base import BaseStream


class GenericStream[T](BaseStream[T]):
    
    def select[V](self, selector:Callable[[T], V]):
        for d in self:
            yield selector(d)
    
    def filter(self, *filters:Callable[[T], bool]):
        for d in self:
            if not all(f(d) for f in filters):
                continue

            yield d
    
    def unwind[V](self, roller:Callable[[T], Iterable[V]]):
        for d in self:
            yield from roller(d)
    
    def sort(self, key:Callable[[T], Any], reverse=False):
        return sorted(self, key=key, reverse=reverse)

    def take(self, n:int):
        for i, d in enumerate(self):
            if i >= n:
                break
            
            yield d

    def groupby[K, V](self, key:Callable[[T], K], value:Callable[[T], V]):
        data = dict[K, list[V]]()
        for d in self:
            data.setdefault(key(d), []).append(value(d))

        yield from data.items()

    @overload
    def max(self, key:Callable[[T], Any], default:None=None) -> T|None:...
    
    @overload
    def max(self, key:Callable[[T], Any], default:Any) -> T:...
    
    def max(self, key:Callable[[T], Any], default:T|None=None):
        return max(self, key=key, default=default)
    
    @overload
    def min(self, key:Callable[[T], Any], default:None=None) -> T|None:...
    
    @overload
    def min(self, key:Callable[[T], Any], default:Any) -> T:...
    
    def min(self, key:Callable[[T], Any], default:T|None=None):
        return min(self, key=key, default=default)

    def accumulate[D, S](
            self,
            intial_state:S, 
            selector:Callable[[T], D],
            accumulator:Callable[[S, D], S|None],
            stateful=True
        ):
        state = intial_state
        for d in self:
            projected = selector(d)
            new_state = accumulator(state, projected)
            if new_state is not None and not stateful:
                state = new_state

        return state

    def sample(self, n:int, seed:int|None=None, replace=False):
        data = self._data
        if not isinstance(data, Sequence):
            data = [*self]
        
        rng = np.random.default_rng(seed)
        args = rng.choice(len(data), size=n, replace=replace)
        
        for a in args:
            yield cast(T, data[a])

