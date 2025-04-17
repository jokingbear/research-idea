from ...functional import AutoPipe


class FlowExceptionHandler(AutoPipe):
    
    def run(self, block:str, data, e:Exception):
        raise e
