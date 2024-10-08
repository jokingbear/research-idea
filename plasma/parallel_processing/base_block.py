from warnings import warn


class Block:

    def __init__(self, inputs, outputs):
        warn('this class is deprecated and will be removed in the future, please use communicators and queues')
        self.inputs = inputs
        self.outputs = outputs

    def init(self):
        pass

    def terminate(self, exc_type, exc_val, exc_tb):
        pass

    def __enter__(self):
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate(exc_type, exc_val, exc_tb)
