import os
import torch


def set_devices(*device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device_ids])


class eval:

    def __init__(self, *models:torch.nn.Module) -> None:
        self.models = [m.eval() for m in models]
        self._no_grad = torch.no_grad()
    
    def __enter__(self):
        self._no_grad.__enter__()
        return self.models

    def __exit__(self, *_):
        self._no_grad.__exit__(*_)
