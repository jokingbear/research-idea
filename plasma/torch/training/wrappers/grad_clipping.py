import torch

from ..prototypes.trainer_wrapper import TrainerWrapper


class GradientClipping(TrainerWrapper):

    def __init__(self, max_norm=1, norm_type=2):
        super().__init__()

        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def chain(self, trainer, i, inputs, outputs):
        model = trainer.model

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 
            max_norm=self.max_norm,
            norm_type=self.norm_type, 
            error_if_nonfinite=False,
        )
