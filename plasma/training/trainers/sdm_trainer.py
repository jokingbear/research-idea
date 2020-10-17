from typing import Tuple

import torch.nn as nn

from .base_trainer import BaseTrainer
from .utils import *


class SDM(BaseTrainer):

    def __init__(self, model, optimizer, loss=None, alpha=1, device=None):
        loss = loss or nn.MSELoss()
        super().__init__([model], [optimizer], loss)

        self.alpha = alpha
        self.device = device

    def _extract_data(self, batch_data):
        return get_batch_tensors(batch_data, "float", self.device, "float", self.device)

    def _train_one_batch(self, inputs, targets) -> Tuple[dict, object]:
        inputs = {
            "student_inputs": inputs,
            "teacher_inputs": targets
        }

        outputs = self.models[0](**inputs)
        student_features = outputs["student_features"]
        student_projects = outputs["student_projects"]
        student_cycle = outputs["student_cycle"]
        student_intermediates = outputs["student_intermediates"]

        teacher_features = outputs["teacher_features"]
        teacher_projects = outputs["teacher_projects"]
        teacher_cycle = outputs["teacher_cycle"]
        teacher_intermediates = outputs["teacher_intermediates"]

        student_proj_loss = self.loss(student_projects, teacher_features)
        student_cycle_loss = self.loss(student_cycle, student_features)

        teacher_proj_loss = self.loss(teacher_projects, student_features)
        teacher_cycle_loss = self.loss(teacher_cycle, teacher_features)

        int_loss = 0
        for student_int, teacher_int in zip(student_intermediates, teacher_intermediates):
            int_loss = int_loss + self.loss(student_int, teacher_int)

        total_loss = student_proj_loss + student_cycle_loss + teacher_proj_loss + teacher_cycle_loss
        total_loss = total_loss + self.alpha * int_loss
        total_loss.backward()

        self.optimizers[0].step()

        return {
            "student_proj": float(student_proj_loss),
            "student_cycl": float(student_cycle_loss),
            "teacher_proj": float(teacher_proj_loss),
            "teacher_cycl": float(teacher_cycle_loss),
            "hint": float(int_loss),
            "loss": float(total_loss),
        }, None

    def _get_train_measures(self, inputs, targets, loss_dict, cache) -> dict:
        return loss_dict

    def _get_eval_cache(self, inputs, targets):
        return None

    def _get_eval_logs(self, eval_caches) -> dict:
        return {}

    def extra_repr(self):
        return f"alpha={self.alpha}, device={self.device}"
