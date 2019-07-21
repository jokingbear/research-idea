from keras import layers, Sequential, Model, datasets as dts, backend as K, optimizers as opts

import numpy as np
import matplotlib.pyplot as plt


def discriminator(input_shape=(28, 28, 1)):
    arch = [
        layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(negative_slope=0.2),
        layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.ReLU(negative_slope=0.2),
        layers.Conv2D(64, 3, padding="same", strides=2, kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.ReLU(negative_slope=0.2),
        layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.ReLU(negative_slope=0.2),
        layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.ReLU(negative_slope=0.2),
        layers.Conv2D(128, 3, padding="same", strides=2, kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.ReLU(negative_slope=0.2),
        layers.Flatten(),
        layers.Dense(1)
    ]

    model = Sequential(arch)

    return model


def generator(z_shape=128):
    arch = [
        layers.Dense(7*7*256, kernel_initializer="he_normal", input_shape=(z_shape,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape([7, 7, 256], input_shape=(z_shape,)),
        layers.Conv2DTranspose(128, 3, strides=2, padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, 3, strides=2, padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(1, 3, padding="same", activation="tanh")
    ]

    model = Sequential(arch)

    return model


def difference_model(d):
    train_img = layers.Input(shape=d.input_shape[1:])
    anchor_img = layers.Input(shape=d.input_shape[1:])

    train_c = d(train_img)
    anchor_c = d(anchor_img)

    diff = layers.Lambda(lambda args: args[0] - K.mean(args[1]))([train_c, anchor_c])

    output = layers.Activation("sigmoid")(diff)

    return Model([train_img, anchor_img], output)


def combine_model(d, g):
    d.trainable = False
    z = layers.Input((g.input_shape[-1],))
    img = layers.Input(d.input_shape[1:])

    gz = g(z)

    diff = difference_model(d)
    diff1 = diff([gz, img])
    diff2 = diff([img, gz])
    
    return Model([z, img], diff1), Model([img, z], diff2)


opt = opts.adam(lr=2E-4, beta_1=0.5)

d = discriminator()
g = generator(512)
d_freeze = discriminator()

diff = difference_model(d)
combine1, combine2 = combine_model(d_freeze, g)

diff.compile(opt, loss="binary_crossentropy")
combine1.compile(opt, loss="binary_crossentropy")
combine2.compile(opt, loss="binary_crossentropy")

(x_train, _), _ = dts.mnist.load_data()

train = x_train[..., None] / 127.5 - 1
batch_size = 32
epochs = 200

mean = np.zeros(shape=(512,))
std = np.identity(512)
real = np.ones(shape=batch_size//2)
fake = np.zeros(shape=batch_size//2)

for i in range(epochs):
    for j in range(train.shape[0] // batch_size):
        idc = np.random.choice(train.shape[0], size=batch_size//2, replace=False)
        z = np.random.multivariate_normal(mean, std, size=batch_size//2)
        
        real_img = train[idc]
        fake_img = g.predict(z)

        img = np.concatenate([real_img, fake_img], axis=0)
        diff.train_on_batch([real_img, fake_img], real)
        diff.train_on_batch([fake_img, real_img], fake)

        d_freeze.set_weights(d.get_weights())

        idc = np.random.choice(train.shape[0], size=batch_size//2, replace=False)

        combine1.train_on_batch([z, real_img], real)
        combine2.train_on_batch([real_img, z], fake)

        print(f"finish iteration {j}")

        if j % 50 == 0:
            z = np.random.multivariate_normal(mean, std, size=4)
            imgs = g.predict(z)

            _, f = plt.subplots(ncols=4)

            for i in range(4):
                f[i].imshow(imgs[i, ..., 0])

            plt.show()
