import torch
import numpy as np

from torch import nn, optim as opts
from plasma.training import Trainer, metrics, callbacks, data, utils
from plasma.modules import *
from tensorflow.keras.datasets import mnist

utils.on_notebook = False

(x, y), _ = mnist.load_data()

x = x[y < 3] / 127.5 - 1
y = y[y < 3]


class Data(data.StandardDataset):
    
    def get_len(self):
        return x.shape[0]
    
    def get_item(self, idx):
        return x[idx, None], y[idx]


model = nn.Sequential(*[
    PrimaryGroupConv2d(1, 8, kernel_size=7, stride=2, padding=3),
    GroupBatchNorm2d(8),
    nn.ReLU(inplace=True),
    GroupConv2d(8, 16, kernel_size=7, stride=2, padding=3),
    GroupBatchNorm2d(16),
    nn.ReLU(inplace=True),
    GroupGlobalAverage(),
    nn.Linear(16 * 12, 3)
])

model.cuda(0)
loss = nn.CrossEntropyLoss()

opt = opts.SGD(model.parameters(), lr=0.25, momentum=0.9, nesterov=True)
trainer = Trainer(model, opt, loss, metrics=[metrics.accuracy], x_device="cuda:0", y_device="cuda:0")

cbs = [
    # callbacks.LrFinder(min_lr=1e-5, max_lr=2, epochs=3)
    callbacks.WarmRestart(1e-5, 10, factor=2, cycles=1, snapshot=False),
    # callbacks.CLR(1e-5, 4),
    # callbacks.TrainingScheduler(epochs=1)
    callbacks.CSVLogger("train.csv")
]

trainer.fit(Data(), callbacks=cbs, batch_size=64)

import matplotlib.pyplot as plt

a = Data()[2][0]
ar = np.rot90(a, axes=[1, 2])
ar = np.copy(ar)

_, ax = plt.subplots(ncols=2)
ax[0].imshow(a[0])
ax[1].imshow(ar[0])
plt.show()

from albumentations import Rotate

img = torch.tensor(a[None, None], dtype=torch.float, device="cuda:0")
print(model.eval()(img))

ar30 = Rotate(limit=(-30, -30), always_apply=True, border_mode=1)(image=ar[0])["image"]
plt.imshow(ar30)
plt.show()

img = torch.tensor(ar30[None, None], dtype=torch.float, device="cuda:0")
print(model.eval()(img))
