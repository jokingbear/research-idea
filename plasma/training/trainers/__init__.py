from .standard_trainer import StandardTrainer as Trainer
from .sam_trainer import SAM
from .base_trainer import BaseTrainer


__mapping__ = {
    "standard": Trainer,
    'sam': SAM,
}
