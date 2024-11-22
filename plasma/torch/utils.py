import os
import torch


def set_devices(*device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device_ids])


def count_parameters(model:torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
