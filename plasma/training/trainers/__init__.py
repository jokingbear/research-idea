from .standard_trainer import StandardTrainer as Trainer
from .asam_trainer import ASAM
from .base_trainer import BaseTrainer


__mapping__ = {
    "standard": Trainer,
    'asam': ASAM,
}
