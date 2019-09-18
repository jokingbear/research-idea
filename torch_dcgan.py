import torch
import torch.nn as nn
import torch.optim as opts
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import datasets as dts
from torch_modules import common_modules as commons

(x_train, _), _ = dts.mnist.load_data()

d = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2),
    commons.Reshape(7 * 7 * 256),
    nn.Linear(7 * 7 * 256, 1),
    nn.Sigmoid()
)

g = nn.Sequential(
    nn.Linear(128, 7 * 7 * 256),
    nn.BatchNorm2d(7 * 7 * 256),
    nn.ReLU(),
    commons.Reshape(256, 7, 7),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 1, 1),
    nn.Tanh()
)

d_opt = opts.Adam(d.parameters(), lr=2E-4, betas=(0.5, 0.999))
g_opt = opts.Adam(g.parameters(), lr=2E-4, betas=(0.5, 0.999))

loss1 = nn.BCELoss()
loss2 = nn.BCELoss()

epochs = 100
batch_size = 32

n_iter = x_train.shape[0] // batch_size
mean = np.array([0] * 128)
var = np.identity(128)

is_real = torch.ones(batch_size)
is_fake = torch.ones() - is_real

for _ in range(epochs):
    for i in range(n_iter):
        idc = np.random.choice(n_iter, size=batch_size, replace=True)
        imgs = x_train[idc, np.newaxis, ...] / 127.5 - 1

        imgs = torch.tensor(imgs)
        z = np.random.multivariate_normal(mean, var, size=batch_size)

        d_opt.zero_grad()

        d = d.train()
        g = g.eval()

        reals = d(imgs)
        d_loss = loss1(reals, is_real)
        d_loss.backward()
        d_opt.step()

        d_opt.zero_grad()
        fakes = d(g(z))
        d_loss = loss1(fakes, is_fake)
        d_loss.backward()
        d_opt.step()

        d = d.eval()
        g = g.train()

        g_opt.zero_grad()
        reals = d(g(z))
        g_loss = loss2(reals, is_real)
        g_loss.backward()
        g_opt.step()

        if i % 50 == 0:
            z = np.random.multivariate_normal(mean, var, size=batch_size)
            g = g.eval()
            gimgs = g(z).numpy()

            _, f = plt.subplots(ncols=4, figsize=(15, 15))

            [f[i].imshow(gimgs[i, 0]) for i in range(4)]
