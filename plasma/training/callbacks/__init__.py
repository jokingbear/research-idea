from .base_class import Callback
from .standard_callbacks import CSVLogger, Tensorboard
from .standard_callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from .clr import SuperConvergence, LrFinder, WarmRestart


__mapping__ = {
    "csv logger": CSVLogger,
    "csv_logger": CSVLogger,
    "csv": CSVLogger,
    "tensorboard": Tensorboard,
    "plateau": ReduceLROnPlateau,
    "early stopping": EarlyStopping,
    "early_stopping": EarlyStopping,
    "checkpoint": ModelCheckpoint,
    "super convergence": SuperConvergence,
    "super_convergence": SuperConvergence,
    "lr finder": LrFinder,
    "lr_finder": LrFinder,
    "warm": WarmRestart,
    "warm restart": WarmRestart,
    "warm_restart": WarmRestart,
}
