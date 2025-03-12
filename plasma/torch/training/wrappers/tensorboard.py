from torch.utils.tensorboard import SummaryWriter
from ..bases import ForwardWrapper


class Tensorboard(ForwardWrapper):

    def __init__(self, log_dir):
        super().__init__()

        self.log_dir = log_dir
        self._writer = SummaryWriter(self.log_dir)
        self._counter = 0

    def append(self, trainer, i, inputs, outputs):
        writer = self._writer
        loss_val = outputs
        step = self._counter

        writer.add_scalar('loss', loss_val.float(), step)
        if trainer.scheduler is not None:
            for i, lr in enumerate(trainer.scheduler.get_last_lr()):
                writer.add_scalar(f'lr-{i}', lr, step)

        self._counter += 1
