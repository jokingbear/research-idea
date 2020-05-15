import numpy as np


def _assert_inputs(pred, true):
    assert pred.shape == true.shape, f"predition shape {pred.shape} is not the same as label shape {true.shape}"


def get_class_balance_weight(counts):
    total = counts.values[0, 0] + counts.values[0, 1]
    beta = 1 - 1 / total

    weights = (1 - beta) / (1 - beta ** counts)
    normalized_weights = weights / weights.values[:, 0, np.newaxis]

    return normalized_weights
