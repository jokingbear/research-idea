import re

from abc import abstractmethod


class Pipe:

    def __init__(self, **kwargs):
        self._marked_attributes = []

        for attr, val in kwargs.items():
            self._marked_attributes.append(attr)
            setattr(self, attr, val)

    @abstractmethod
    def run(self, *inputs, **kwargs):
        pass

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
        rep = re.sub('\([\t\n]{1,}\)', '()', rep)
        return f'{type(self).__name__}({rep})'
