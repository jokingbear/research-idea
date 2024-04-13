import re

from abc import abstractmethod
from .utils import partials


class Pipe:

    def __init__(self, **kwargs):
        self._marked_attributes = []
        self._hooks = []

        for attr, val in kwargs.items():
            self._marked_attributes.append(attr)
            setattr(self, attr, val)

    @abstractmethod
    def run(self, *inputs, **kwargs):
        pass
    
    def add_logger(self, logging_func):
        self._hooks.append(logging_func)

        self.run = _HookRunner(self)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __repr__(self):
        rep = []
        for attr in self._marked_attributes:
            val = getattr(self, attr)
            val_rep = repr(val)
            lines_rep = val_rep.split('\n')

            if len(lines_rep) == 1:
                rep.append(f'\t{attr}={lines_rep[0]},\n')
            elif len(lines_rep) > 1:
                body = []
                for line in lines_rep[1:-1]:
                    body.append('\t' + line + '\n')
                body = ''.join(body)
                rep.append(f'\t{attr}={lines_rep[0]}\n{body}\t{lines_rep[-1]},\n')

        rep = ''.join(rep)
        rep = '\n' + rep
        rep = re.sub(r'\([\t\n]{1,}\)', '()', rep)
        return f'{type(self).__name__}({rep})'


class _HookRunner:

    def __init__(self, pipe: Pipe) -> None:
        self._pipe = pipe
        self._original_func = pipe.run
    
    def __call__(self, *args, **kwargs):
        inputs = {
            'args': args,
            'kwargs': kwargs
        }

        try:
            outputs = self._original_func(*args, **kwargs)
        except Exception as e:
            outputs = e

        for logger in self._pipe._hooks:
            logger(inputs, outputs)

        if isinstance(outputs, Exception):
            raise outputs

        return outputs
