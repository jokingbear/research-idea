from torch.utils.tensorboard import SummaryWriter
from ..bases import ForwardWrapper


class Tensorboard(ForwardWrapper):

    def __init__(self, log_dir, n=1):
        super().__init__()

        self.log_dir = log_dir
        self._writer = SummaryWriter(self.log_dir)
        self._counter = 0
        self.n = n

    def append(self, trainer, i, inputs, outputs):
        if trainer.rank == 0 and outputs is not None:
            self.log(trainer, self._counter, inputs, outputs)
        
        self._counter += 1

    def log(self, trainer, step, inputs, outputs):
        writer = self._writer
        writer.add_scalar('loss', outputs.float(), step)
        if trainer.scheduler is not None:
            for i, lr in enumerate(trainer.scheduler.get_last_lr()):
                writer.add_scalar(f'lr-{i}', lr, step)
