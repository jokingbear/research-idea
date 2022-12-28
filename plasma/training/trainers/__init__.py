from .standard_trainer import StandardTrainer as Trainer
from .deepspeed_trainer import DeepspeedTrainer


__mapping__ = {
    "standard": Trainer,
    'deepspeed': DeepspeedTrainer,
}
