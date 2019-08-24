import tensorflow as tf

from tensorflow.keras import layers
from implementations import group_conv as gc


con_layer = layers.Conv2D
decon_layer = layers.Conv2DTranspose
group_layer = gc.GroupConv2D
routing_layer = gc.GroupRouting2D
pooling_layer = layers.MaxPool2D


def activation_layer(x): return layers.ReLU(negative_slope=0.2)(x)


def con_block(x, f, kernel=3, stride=1, relu=True, normalization=None):
    con = con_layer(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal")(x)
    con = normalization(con) if normalization else con
    con = activation_layer(con) if relu else con

    return con


def decon_block(x, f, kernel=3, stride=2, relu=True, normalization=None):
    decon = decon_layer(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal")(x)
    decon = normalization(decon) if normalization else decon
    decon = activation_layer(decon) if relu else decon

    return decon


def group_block(x, n_group, f, kernel=3, stride=1, dilation=1, relu=True, normalization=None):
    con = group_layer(n_group, f, kernel, strides=stride, dilation_rate=dilation, kernel_initializer="he_normal")(x)
    con = normalization(con) if normalization else con
    con = activation_layer(con) if relu else con

    return con


def routing_block(x, n_group, f, kernel=3, stride=1, relu=True, normalization=None, n_iter=3):
    con = routing_layer(n_group, f, kernel, strides=stride, n_iter=n_iter)(x)
    con = normalization(con) if normalization else con
    con = activation_layer(con) if relu else con

    return con


def res_block(x, n_group, bottleneck, n_iter=3, down_sample=False, normalizations=(None, None, None)):
    f = int(x.shape[-1])

    con = con_block(x, n_group * bottleneck, kernel=1, normalization=normalizations[0])
    con = group_block(con, n_group, bottleneck, stride=2 if down_sample else 1, normalization=normalizations[1])

    if n_iter == 1:
        con = con_block(con, f, kernel=1, relu=down_sample, normalization=normalizations[-1])
    else:
        con = routing_block(con, n_group, f, kernel=1, relu=down_sample, normalization=normalizations[-1],
                            n_iter=n_iter)

    x = pooling_layer()(x) if down_sample else x
    res = layers.concatenate([x, con]) if down_sample else layers.add([x, con])
    res = res if down_sample else activation_layer(res)

    return res


def scale_res_block(x, n_group, bottleneck, n_iter=3, down_sample=False, normalizations=(None, None, None)):
    f = int(x.shape[-1])
    g = n_group
    b = bottleneck

    spatial_shape = x.shape[1:-1]

    con = con_block(x, g * b, kernel=1, normalization=normalizations[0])
    cons = tf.reshape(con, [-1] + spatial_shape + [4, g * b // 4])
    con1, con2, con3, con4 = [cons[..., i, :] for i in range(4)]

    grp1 = group_block(con1, g // 4, b, stride=2 if down_sample else 1, normalization=normalizations[1])
    grp2 = group_block(con2, g // 4, b, stride=2 if down_sample else 1, normalization=normalizations[2], dilation=6)
    grp3 = group_block(con3, g // 4, b, stride=2 if down_sample else 1, normalization=normalizations[3], dilation=12)
    grp4 = group_block(con4, g // 4, b, stride=2 if down_sample else 1, normalization=normalizations[4], dilation=18)

    if n_iter == 1:
        grp1 = con_block(grp1, f // 4, kernel=1, relu=False)
        grp2 = con_block(grp2, f // 4, kernel=1, relu=False)
        grp3 = con_block(grp3, f // 4, kernel=1, relu=False)
        grp4 = con_block(grp4, f // 4, kernel=1, relu=False)
    else:
        grp1 = routing_block(grp1, g // 4, f // 4, kernel=1, relu=False, n_iter=n_iter)
        grp2 = routing_block(grp2, g // 4, f // 4, kernel=1, relu=False, n_iter=n_iter)
        grp3 = routing_block(grp3, g // 4, f // 4, kernel=1, relu=False, n_iter=n_iter)
        grp4 = routing_block(grp4, g // 4, f // 4, kernel=1, relu=False, n_iter=n_iter)

    grp = layers.concatenate([grp1, grp2, grp3, grp4])
    grp = normalizations[-1](grp)
    grp = activation_layer(grp)

    x = pooling_layer()(x) if down_sample else x
    res = layers.concatenate([x, grp])

    return res


def cse_block(x):
    c = int(x.shape[-1])

    d = layers.GlobalAvgPool2D()(x)
    d = layers.Dense(c // 2, kernel_initializer="he_normal")(d)
    d = activation_layer(d)
    d = layers.Dense(c, activation="sigmoid")(d)

    return d * x


def sse_block(x):
    con = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return con * x
