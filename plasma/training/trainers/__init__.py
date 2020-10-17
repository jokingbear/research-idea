from .standard_trainer import StandardTrainer as Trainer
from .base_trainer import BaseTrainer
from .sdm_trainer import SDM


__mapping__ = {
    "standard": Trainer,
    "sdm": SDM,
}
