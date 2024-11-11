from torch.utils.tensorboard import SummaryWriter
from ..prototypes.trainer_wrapper import TrainerWrapper


class Tensorboard(TrainerWrapper):

    def __init__(self, log_dir):
        super().__init__()

        self.log_dir = log_dir
        self._writer = SummaryWriter(self.log_dir)
        self._counter = 0

    def chain(self, trainer, i, inputs, outputs):
        writer = self._writer
        epoch = trainer.current_epoch
        loss_val = outputs

        if self._counter > i:
            step = epoch * self._counter + i
        else:
            step = self._counter = i

        writer.add_scalar('loss', loss_val.float(), step)
        if trainer.scheduler is not None:
            for i, lr in enumerate(trainer.scheduler.get_last_lr()):
                writer.add_scalar(f'lr-{i}', lr, step)
