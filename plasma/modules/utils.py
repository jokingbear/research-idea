import torch.nn as nn


def walkthrough(m: nn.Module, diving_func):
    for name, child in m.named_children():
        diving = diving_func(m, name, child)
        if diving:
            walkthrough(child, diving_func)
