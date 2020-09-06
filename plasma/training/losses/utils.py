import numpy as np


def _assert_inputs(pred, true):
    assert pred.shape == true.shape, f"predition shape {pred.shape} is not the same as label shape {true.shape}"
