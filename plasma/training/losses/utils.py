import numpy as np


def _assert_inputs(pred, true):
    assert pred.shape == true.shape, f"predition shape {pred.shape} is not the same as label shape {true.shape}"


def get_class_balance_weight(counts, anchor=0):
    """
    calculate class balance weight from counts with anchor
    :param counts: class counts, shape=(n_class, 2)
    :param anchor: make anchor class weight = 1 and keep the aspect ratio of other weight
    :return: weights for cross entropy loss
    """
    total = counts.values[0, 0] + counts.values[0, 1]
    beta = 1 - 1 / total

    weights = (1 - beta) / (1 - beta ** counts)
    normalized_weights = weights / weights.values[:, anchor, np.newaxis]

    return normalized_weights
