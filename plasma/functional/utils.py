from abc import ABC, abstractmethod
from typing import Any
from .pipes import SequentialPipe


class proxy_func(ABC):

    def __init__(self, func) -> None:
        self.func = func

        if hasattr(func, '__qualname__'):
            self.__qualname__ = func.__qualname__

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __get__(self, instance, owner):
        def run_instance(*args, **kwargs):
            return self.__call__(instance, *args, **kwargs)
        return run_instance


class auto_map_func(proxy_func):

    def __call__(self, inputs):
        if isinstance(inputs, (tuple, list)):
            return self.func(*inputs)
        elif isinstance(inputs, dict):
            return self.func(**inputs)
        elif inputs is None:
            return self.func()
        else:
            return self.func(inputs)


class partials(proxy_func):

    def __init__(self, func, *args, pre_apply_before=True, **kwargs):
        super().__init__(func)
        self.args = args
        self.kwargs = kwargs
        self.pre_apply_before = pre_apply_before

    def __call__(self, *new_args, **new_kwargs):
        if self.pre_apply_before:
            return self.func(*self.args, *new_args, **self.kwargs, **new_kwargs)
        else:
            return self.func(*new_args, *self.args, **new_kwargs, **self.kwargs)

    def __repr__(self):
        func_repr = self.func.__name__

        params = []
        args = [str(a) for a in self.args]
        params += args

        if self.pre_apply_before:
            params.append('*')
        else:
            params = ['*'] + params

        kwargs = [f'{k}={v}' for k, v in self.kwargs.items()]
        if self.pre_apply_before:
            params += kwargs
            params.append('**')
        else:
            params += ['**'] + kwargs

        params = ','.join(params)

        return f'{func_repr}({params})'


class chain(SequentialPipe):

    def __init__(self, func1, func2) -> None:
        super().__init__()

        self.func1 = func1
        self.func2 = func2

    def __get__(self, instance, owner):
        return partials(self.run, instance)
