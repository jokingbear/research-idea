import re

from abc import abstractmethod


class AutoPipe[T]:

    def __init__(self):
        self._marked_attributes = []

    @abstractmethod
    def run(self, *inputs, **kwargs) -> T:
        pass
    
    def __setattr__(self, key:str, value):
        if key[0] != '_' and key not in self._marked_attributes:
            self._marked_attributes.append(key)

        super().__setattr__(key, value)

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
        if len(rep) > 0:
            rep = '\n' + rep
            rep = re.sub(r'\([\t\n]{1,}\)', '()', rep)
        return f'{type(self).__name__}({rep})'
