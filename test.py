import torch
import torch.nn as nn
import torch.optim as opts
import numpy as np

from torch_modules import commons
from torch_modules.training import Trainer, data, callbacks


model = nn.Sequential(*[
    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
    commons.GlobalAverage(),
    nn.Linear(32, 10),
    nn.Softmax()
])

i = 0


def metric(y_true, y_pred):
    global i
    i -= 1
    return i


model = model.cuda(0)
loss = nn.CrossEntropyLoss()
loss_fn = lambda yt, yp: loss(yp, yt)
opt = opts.Adam(model.parameters())
trainer = Trainer(model, opt, loss_fn, metrics=[metric])


class Data(data.Sequence):

    def get_len(self):
        return 8

    def get_item(self, idx):
        return np.random.rand(16, 1, 8, 8), np.random.choice(10, size=16)

    def shuffle(self):
        pass


d = Data(x_gpu="cuda:0", y_gpu="cuda:0")

cbs = [
    callbacks.ReduceLROnPlateau(monitor="val_metric", mode="max"),
    callbacks.CSVLogger("tmp.csv")
]

trainer.fit(d, d, epochs=15, callbacks=cbs)
