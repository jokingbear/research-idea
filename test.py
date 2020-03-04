import torch
import torch.nn.functional as func

from plasma.training import callbacks
from tensorflow.keras.datasets import mnist
from torchvision.utils import save_image

(x_train, y_train), _ = mnist.load_data()


a = torch.tensor(x_train[:64], dtype=torch.float)

b = torch.tensor(y_train[:64], dtype=torch.long)
b = func.one_hot(b, num_classes=10).type(torch.float)

cutmix = callbacks.augmentations.CutMix()

cutmix.on_training_batch_begin(0, a, b)
save_image(a[:, None, ...], "images.png", normalize=True)
