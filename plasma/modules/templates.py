import torch.nn as nn


def classifier(backbone, head, preprocess=None):
    """
    create a classification Sequential module
    Args:
        backbone: backbone with outputs, [N, features]
        head: classification head
        preprocess: preprocess module before backbone

    Returns: Sequential module
    """
    model = nn.Sequential()

    if preprocess is not None:
        model.preprocess = preprocess

    model.features = backbone
    model.head = head

    return model
