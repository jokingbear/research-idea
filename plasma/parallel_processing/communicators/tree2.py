from .tree import TreeFlow
from ..queues import Queue
from ...functional import partials
from ._proxy import ProxyIO
from .handler import FlowExceptionHandler


class StableTree(TreeFlow):
    
    def __init__(self):
        super().__init__()
        
        self._exception_handler = None
    
    def on_exception(self, handler:FlowExceptionHandler):
        assert not self.running, \
            'tree is already running, please release it to register new exception handler'
        assert handler is not None, 'handler can not be None'

        self._exception_handler = handler

    def run(self):
        for n, q in self.queues.items():
            if n is not ProxyIO:
                assert q is not None, f'no queue registered for block {n}'
                exception_handler = self._exception_handler
                if exception_handler is not None:
                    exception_handler = partials(exception_handler, n)
                    q.on_exception(exception_handler)

        return super().run()
    
    @property
    def queues(self) -> dict[str, Queue]:
        return {n: attrs.get('queue', None) for n, attrs in self._module_graph.nodes.items()}

    def is_alive(self):
        return self.running and all(q.is_alive() for q in self.queues.values() if q is not None)
