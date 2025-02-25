import torch

from ..bases.trainer_wrapper import TrainerWrapper


class GradientClipping(TrainerWrapper):

    def __init__(self, max_norm=1, norm_type=2):
        super().__init__()

        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def backward(self, trainer, i, inputs, obj_val):
        if trainer.scaler is not None:
            trainer.scaler.unscale_(trainer.optimizer)

        model = trainer.model

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 
            max_norm=self.max_norm,
            norm_type=self.norm_type, 
            error_if_nonfinite=False,
        )
