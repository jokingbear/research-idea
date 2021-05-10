from torch.nn import BCELoss, MSELoss, L1Loss
from .standard_losses import CombineLoss, FbetaLoss, FocalLoss, WBCE

__mapping__ = {
    "wbce": WBCE,
    "focal": FocalLoss,
    "fb": FbetaLoss,
    "bce": BCELoss,
    "mse": MSELoss,
    "l2": MSELoss,
    "l1": L1Loss,
}
