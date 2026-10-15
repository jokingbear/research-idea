from itertools import chain, product
from typing import Any
from collections.abc import Callable, Iterable, Sequence

from .generic import GenericStream
from ....functional import auto_map


class Stream[T](GenericStream[T]):
    
    def select[V](self, selector: Callable[[T], V]): # type: ignore
        return Stream(super().select(selector))

    def filter(self, *filters:Callable[[T], bool]): # type: ignore
        return Stream(super().filter(*filters))
                 
    def unwind[V](self, roller:Callable[[T], Iterable[V]]): # type: ignore
        return Stream(super().unwind(roller))
    
    def zip_unwind[*V](self, roller:Callable[[T], Iterable[tuple[*V]]]):
        return ZippedStream(super().unwind(roller))
    
    def sort(self, key:Callable[[T], Any], reverse=False): # type: ignore
        return Stream(super().sort(key, reverse))
    
    def take(self, n:int): # type: ignore
        return Stream(super().take(n))
    
    def groupby[K, V](self, key:Callable[[T], K], value:Callable[[T], V]): # type: ignore
        return GroupStream(super().groupby(key, value))

    def split[*V](self, splitter:Callable[[T], tuple[*V]]):
        return ZippedStream(splitter(d) for d in self)
    
    def enumerate(self, start=0):
        return ZippedStream(enumerate(self, start=start))
    
    def product[V](self, data:Iterable[V]):
        return ZippedStream((d1, d2) for d1, d2 in product(self, data))
    
    def zip_product[*V](self, data:Iterable[tuple[*V]]):
        return ZippedStream((d1, *d2) for d1, d2 in product(self, data))
    
    def sample(self, n: int, seed: int | None = None, replace=False): # type: ignore
        return Stream(super().sample(n, seed, replace))
    
    def evaluate(self):
        return list(self)
    
    @staticmethod
    def from_iterable[K](iterable:Iterable[Iterable[K]]):
        return Stream(chain().from_iterable(iterable))
    
    @staticmethod
    def chain[K](*data:Iterable[K]):
        return Stream(chain(*data))


class GroupStream[K, V](GenericStream[tuple[K, Sequence[V]]]):

    def collect[O](self, collector:Callable[[K, Sequence[V]], O]):
        return Stream(super().select(lambda kv: collector(*kv)))

    def map_key[K2](self, mapper:Callable[[K], K2]):
        return GroupStream(super().select(lambda kv: (mapper(kv[0]), kv[1])))
    
    def map_value[T](self, applier:Callable[[K, Sequence[V]], Iterable[T]]):
        return GroupStream(super().select(lambda kv: (kv[0], [*applier(*kv)])))
    
    def select[T](self, selector:Callable[[K, Sequence[V]], T]): # type: ignore
        return ZippedStream(super().select(lambda kv: (kv[0], selector(*kv))))
    
    def zip_select[*T](self, selector:Callable[[K, Sequence[V]], tuple[*T]]):
        return ZippedStream((k, *r) for k, r in self.select(selector))
    
    def project[T](self, projector:Callable[[K, Sequence[V]], T]):
        return Stream(projector(k, v) for k, v in self)
    
    def zip_project[*T](self, projector:Callable[[K, Sequence[V]], tuple[*T]]):
        return ZippedStream(projector(k, v) for k, v in self)
    
    def filter(self, *filters:Callable[[K, Sequence[V]], bool]): # type: ignore
        return GroupStream(super().filter(*[auto_map(f) for f in filters]))
         
    def unwind[T](self, roller:Callable[[K, Sequence[V]], Iterable[T]]): # type: ignore
        return Stream(super().unwind(lambda kv:roller(*kv)))

    def zip_unwind[*T](self, roller:Callable[[K, Sequence[V]], Iterable[tuple[*T]]]):
        return ZippedStream(super().unwind(lambda kv:roller(*kv)))

    def sort(self, key: Callable[[K, Sequence[V]], Any], reverse=False): # type: ignore
        return GroupStream(super().sort(lambda kv: key(*kv), reverse))

    def take(self, n: int): # type: ignore
        return GroupStream(super().take(n))

    def groupby[K1, V1](self, key: Callable[[K, Sequence[V]], K1], value: Callable[[K, Sequence[V]], V1]): # type: ignore
        return GroupStream(super().groupby(auto_map(key), auto_map(value)))

    def max(self, key: Callable[[K, Sequence[V]], Any]): # type: ignore
        return super().max(key=lambda kv:key(*kv))
    
    def min(self, key: Callable[[K, Sequence[V]], Any]): # type: ignore
        return super().min(key=lambda kv: key(*kv))

    def evaluate(self):
        return dict(self)

    def accumulate[D, S]( # type: ignore
            self, 
            intial_state:S, selector:Callable[[K, Sequence[V]], D], 
            accumulator:Callable[[S, D], S|None], 
            stateful=True
        ):
        return super().accumulate(intial_state, auto_map(selector), accumulator, stateful)
    
    def sample(self, n: int, seed: int | None = None, replace=False): # type: ignore
        return GroupStream(super().sample(n, seed, replace))


class ZippedStream[*T](GenericStream[tuple[*T]]):
    
    def select[*V](self, selector:Callable[[*T], tuple[*V]]): # type: ignore
        return ZippedStream(super().select(auto_map(selector)))

    def project[V](self, projector:Callable[[*T], V]):
        return Stream(projector(*d) for d in self)

    def filter(self, *filters:Callable[[*T], bool]): # type: ignore
        new_filters = [auto_map(f) for f in filters]
        return ZippedStream(super().filter(*new_filters))
                 
    def unwind[V](self, roller:Callable[[*T], Iterable[V]]): # type: ignore
        return Stream(super().unwind(auto_map(roller)))
    
    def zip_unwind[*V](self, roller:Callable[[*T], Iterable[tuple[*V]]]):
        return ZippedStream(super().unwind(lambda t: roller(*t)))
    
    def sort(self, key:Callable[[*T], Any], reverse=False): # type: ignore
        return ZippedStream(super().sort(lambda d: key(*d), reverse))

    def take(self, n:int): # type: ignore
        return ZippedStream(super().take(n))
    
    def groupby[K, V](self, key:Callable[[*T], K], value:Callable[[*T], V]): # type: ignore
        return GroupStream(super().groupby(auto_map(key), auto_map(value)))

    def max(self, key:Callable[[*T], Any]): # type: ignore
        return super().max(key=lambda t:key(*t))
    
    def min(self, key: Callable[[*T], Any]): # type: ignore
        return super().min(key=lambda t:key(*t))

    def product[V](self, data:Iterable[V]):
        return ZippedStream((*d1, d2) for d1, d2 in product(self, data))
    
    def zip_product[*V](self, data:Iterable[tuple[*V]]):
        return ZippedStream((*d1, *d2) for d1, d2 in product(self, data))

    def enumerate(self, start=0):
        return ZippedStream((i, *d) for i, d in enumerate(self, start))
    
    def evaluate(self):
        return list(self)

    def accumulate[D, S]( # type: ignore
            self, 
            intial_state:S, selector:Callable[[*T], D], 
            accumulator:Callable[[S, D], S|None], 
            stateful=True
        ):
        return super().accumulate(intial_state, auto_map(selector), accumulator, stateful)

    def sample(self, n: int, seed: int | None = None, replace=False): # type: ignore
        return ZippedStream(super().sample(n, seed, replace))
