from abc import abstractmethod


class Pipe:

    def __init__(self, *marked_attributes):
        self._marked_attributes = marked_attributes

    @abstractmethod
    def run(self, *inputs, **kwargs):
        pass

    def __repr__(self):
        rep = []
        for attr in self._marked_attributes:
            val = getattr(self, attr)
            val_rep = repr(val)
            lines_rep = val_rep.split('\n')

            if len(lines_rep) == 1:
                rep.append(f'\t{attr}={lines_rep[0]}\n')
            elif len(lines_rep) > 1:
                body = []
                for line in lines_rep[1:-1]:
                    body.append('\t' + line + '\n')
                body = ''.join(body)
                rep.append(f'\t{attr}={lines_rep[0]}\n{body}\t{lines_rep[-1]}\n')

        rep = ''.join(rep)
        rep = '\n' + rep
        return f'{type(self).__name__}({rep})'
