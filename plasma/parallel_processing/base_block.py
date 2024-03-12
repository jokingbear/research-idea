class Block:

    def __init__(self, inputs, outputs):
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
