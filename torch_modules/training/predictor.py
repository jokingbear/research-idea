import torch


class StandardPredictor:

    def __init__(self, model):
        self.model = model.eval()

    def predict(self, *xs):
        with torch.no_grad():
            xs = [torch.tensor(x) if not torch.is_tensor(x) else x for x in xs]

            return self.model(*xs)
