from .standard_losses import combine_loss, weighted_bce, focal_loss_fn, fb_loss_fn
from .utils import get_class_balance_weight
from torch.nn import BCELoss, MSELoss, L1Loss


__mapping__ = {
    "wbce": weighted_bce,
    "focal loss": focal_loss_fn,
    "focal_loss": focal_loss_fn,
    "fb loss": fb_loss_fn,
    "fb_loss": fb_loss_fn,
    "bce": BCELoss,
    "mse": MSELoss,
    "l2": MSELoss,
    "l1": L1Loss,
}
