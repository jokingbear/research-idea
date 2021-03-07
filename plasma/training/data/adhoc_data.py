from torch.utils import data
from .base_class import StandardDataset


class AdhocData(StandardDataset):

    def __init__(self, arr, mapping=None, imapping=None, kwargs=None):
        super().__init__()

        assert mapping != imapping, "mapping or imapping must be different from each other"

        self.source = arr
        self.mapping = mapping
        self.imapping = imapping
        self.kwargs = kwargs or {}

    def get_len(self):
        return len(self.source)

    def get_item(self, idx):
        item = self.source[idx]
        if self.mapping is not None:
            return self.mapping(item, **self.kwargs)
        else:
            return self.imapping(idx, item, **self.kwargs)
