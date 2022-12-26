from .base_class import StandardDataset


class AdhocData(StandardDataset):

    def __init__(self, arr, mapping, kwargs=None):
        super().__init__()

        self.source = arr
        self.mapping = mapping
        self.kwargs = kwargs or {}

    def get_len(self):
        return len(self.source)

    def get_item(self, idx):
        item = self.source[idx]
        return self.mapping(item, **self.kwargs)
