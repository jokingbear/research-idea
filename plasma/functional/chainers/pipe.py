from typing import Any
from collections.abc import Callable


def identity(x):
    return x


class pipe[I, O]:
    
    def __init__(self, func:Callable[[Any], O], prev:Callable[[I], Any]=identity):
        self.prev = prev
        self.func = func

    def __call__(self, inputs:I) -> O:
        funcs = []
        current = self.prev
        while True:
            if isinstance(current, pipe):
                funcs.append(current.func)
                current = current.prev
            else:
                funcs.append(current)
                break

        funcs = funcs[::-1]
        results = inputs
        for f in funcs:
            results = f(results)
        
        return self.func(results) #type: ignore - lib limit
    
    def chain[T](self, other:Callable[[O], T]):
        return pipe(other, self)

    def partial_left[O1](self, func:Callable[..., O1], *args, **kwargs):
        def alt_func(a:O):
            return func(*args, a, **kwargs)
        
        return pipe(alt_func, self)

    def partial_right[O1](self, func:Callable[..., O1], *args, **kwargs):
        def alt_func(a:O):
            return func(a, *args, **kwargs)
        
        return pipe(alt_func, self)
    
    def automap[O1](self, func:Callable[..., O1]):
        return self.chain(auto_map(func))
    
    def __rshift__[T](self, other:Callable[[O], T]):
        return self.chain(other)

    def func_stack(self):
        p = self
        while isinstance(p, pipe):
            yield p.func
            p = p.prev
        
        yield p
            

class auto_map[O]:
    
    def __init__(self, func:Callable[..., O]):
        if not isinstance(func, pipe):
            func = pipe(func)
            
        self.pipe = func

    def __call__(self, inputs:Any) -> O:
        funcs = [*self.pipe.func_stack()][::-1]
        
        if isinstance(inputs, (tuple, list)):
            results = funcs[1](*inputs)
        elif isinstance(inputs, dict):
            results = funcs[1](**inputs) #type:ignore
        else:
            results = funcs[1](inputs)
        
        for f in funcs[2:]:
            results = f(results)
        return results
    
    def chain[T](self, other:Callable[[O], T]):
        return auto_map(self.pipe.chain(other))

    def partial_left[O1](self, func:Callable[..., O1], *args, **kwargs):
        return auto_map(self.pipe.partial_left(func, *args, **kwargs))

    def partial_right[O1](self, func:Callable[..., O1], *args, **kwargs):
        return auto_map(self.pipe.partial_right(func, *args, **kwargs))
    
    def automap[T](self, other:Callable[..., T]):
        return self.chain(auto_map(other))


class none_propagator[I, O]:
    
    def __init__(self, func:Callable[[I], O]):
        if not isinstance(func, pipe):
            func = pipe(func)
            
        self.pipe:pipe[I, O] = func
    
    def __call__(self, inputs:I|None) -> O|None:
        if inputs is None:
            return
        
        return self.pipe(inputs)
        
    def chain[O1](self, func:Callable[[O], O1]):
        return none_propagator(self.pipe.chain(func))
    
    def automap[O1](self, func:Callable[..., O1]):
        return none_propagator(self.pipe.automap(func))

    def partial_left[O1](self, func:Callable[..., O1], *args, **kwargs):
        return none_propagator(self.pipe.partial_left(func, *args, **kwargs))

    def partial_right[O1](self, func:Callable[..., O1], *args, **kwargs):
        return none_propagator(self.pipe.partial_right(func, *args, **kwargs))
