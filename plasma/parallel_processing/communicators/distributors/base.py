from ....functional import AutoPipe
from ...queues import Queue
from abc import abstractmethod


class Distributor(AutoPipe):

    @abstractmethod
    def run(self, data, *queues:Queue, **named_queues:Queue):
        pass

    def __repr__(self):
        return f'{type(self)}'
