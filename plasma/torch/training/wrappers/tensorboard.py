from torch.utils.tensorboard import SummaryWriter
from ..prototypes.trainer_wrapper import TrainerWrapper


class Tensorboard(TrainerWrapper):

    def __init__(self, log_dir):
        super().__init__()

        self.log_dir = log_dir
        self._writer = SummaryWriter(self.log_dir)
        self._counter = 0

    def chain(self, trainer, state, i, inputs, outputs):
        writer = self._writer
        epoch = state.epoch + 1
        loss_val = outputs

        if self._counter > i:
            step = epoch * self._counter + i
        else:
            step = self._counter = i

        writer.add_scalar('loss', loss_val.float(), step)

        if state.scheduler is not None:
            scheduler = state.scheduler
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
