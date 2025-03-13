import torch

from ..bases import BackwardWrapper


class GradientClipping(BackwardWrapper):

    def __init__(self, max_norm=1, norm_type=2):
        super().__init__()

        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def append(self, trainer, i, inputs, objective_val):
        if objective_val is None:
            return

        if trainer.scaler is not None:
            trainer.scaler.unscale_(trainer.optimizer)

        model = trainer.model

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 
            max_norm=self.max_norm,
            norm_type=self.norm_type, 
            error_if_nonfinite=False,
        )
