from .standard_trainer import StandardTrainer as Trainer


__mapping__ = {
    "standard": Trainer,
    #'deepspeed': DeepspeedTrainer,
}
