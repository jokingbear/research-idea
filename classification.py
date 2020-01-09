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
    nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(inplace=True),
    GlobalAverage(),
    nn.Linear(64, 3)
])

model.cuda(0)
loss = nn.CrossEntropyLoss()

#opt = opts.RMSprop(model.parameters())
opt = opts.SGD(model.parameters(), lr=0.25, momentum=0.9, nesterov=True)
trainer = Trainer(model, opt, loss, metrics=[metrics.accuracy], x_device="cuda:0", y_device="cuda:0")

cbs = [
    #callbacks.LrFinder(min_lr=1e-5, max_lr=2, epochs=3)
    callbacks.WarmRestart(1e-5, 10, factor=2, cycles=3, snapshot=False),
    #callbacks.CLR(1e-5, 4),
    #callbacks.TrainingScheduler(epochs=1)
    callbacks.CSVLogger("train.csv")
]

trainer.fit(Data(), callbacks=cbs, batch_size=64)

cbs[0].plot_data(target="accuracy")