from torch.nn import BCELoss, MSELoss, L1Loss

from .segmentation import TVLoss
from .standard_losses import CombineLoss, FbetaLoss, FocalLoss, WBCE

__mapping__ = {
    "wbce": WBCE,
    "focal loss": FocalLoss,
    "focal_loss": FocalLoss,
    "fb loss": FbetaLoss,
    "fb_loss": FbetaLoss,
    "bce": BCELoss,
    "mse": MSELoss,
    "l2": MSELoss,
    "l1": L1Loss,
    "tv": TVLoss
}
