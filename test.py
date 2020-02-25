from torch.utils.tensorboard import SummaryWriter


a = SummaryWriter("logs")

a.add_custom_scalars()