import numpy as np


def get_class_balance_weight(counts):
    total = counts.values[0, 0] + counts.values[0, 1]
    beta = 1 - 1 / total

    weights = (1 - beta) / (1 - beta ** counts)
    normalized_weights = weights / weights.values[:, 0, np.newaxis]

    return normalized_weights
