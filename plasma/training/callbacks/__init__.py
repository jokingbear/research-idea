from .base_class import Callback
from .standard_callbacks import CSVLogger, Tensorboard
from .standard_callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from .clr import SuperConvergence, LrFinder, WarmRestart


__mapping__ = {
    "csv_logger": CSVLogger,
    "csv": CSVLogger,
    "tensorboard": Tensorboard,
    "reduce_on_plateau": ReduceLROnPlateau,
    "early_stopping": EarlyStopping,
    "checkpoint": ModelCheckpoint,
    "super_convergence": SuperConvergence,
    "lr_finder": LrFinder,
    "warm_restart": WarmRestart,
}
