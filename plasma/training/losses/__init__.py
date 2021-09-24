from torch.nn import BCELoss, MSELoss, L1Loss, CrossEntropyLoss
from .standard_losses import CombineLoss, FbetaLoss, FocalLoss, WBCE

__mapping__ = {
    "wbce": WBCE,
    'ce': CrossEntropyLoss,
    "focal": FocalLoss,
    "fb": FbetaLoss,
    "bce": BCELoss,
    "mse": MSELoss,
    "l2": MSELoss,
    "l1": L1Loss,
}
