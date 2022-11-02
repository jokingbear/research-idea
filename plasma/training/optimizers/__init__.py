from torch.optim import Adam, SGD, AdamW


__mapping__ = {
    "sgd": SGD,
    "adam": Adam,
    'adamw': AdamW,
}
