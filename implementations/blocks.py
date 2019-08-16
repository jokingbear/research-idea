from tensorflow.python.keras import layers
from implementations import group_conv as gc

relu_slope = 0.2

con_layer = layers.Conv2D
decon_layer = layers.Conv2DTranspose
group_layer = gc.GroupConv2D
routing_layer = gc.GroupRouting2D
pooling_layer = layers.MaxPool2D


def con_block(x, f, kernel=3, stride=1, relu=True, normalization=None):
    con = con_layer(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal")(x)
    con = normalization(con) if normalization else con
    con = layers.ReLU(negative_slope=relu_slope)(con) if relu else con

    return con


def decon_block(x, f, kernel=3, stride=2, relu=True, normalization=None):
    decon = decon_layer(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal")(x)
    decon = normalization(decon) if normalization else decon
    decon = layers.ReLU(negative_slope=relu_slope)(decon) if relu else decon

    return decon


def group_block(x, n_group, f, kernel=3, stride=1, relu=True, normalization=None):
    con = group_layer(n_group, f, kernel, strides=stride, kernel_initializer="he_normal")(x)
    con = normalization(con) if normalization else con
    con = layers.ReLU(negative_slope=relu_slope)(con) if relu else con

    return con


def routing_block(x, n_group, f, kernel=3, stride=1, relu=True, normalization=None, n_iter=3):
    con = routing_layer(n_group, f, kernel, strides=stride, n_iter=n_iter)(x)
    con = normalization(con) if normalization else con
    con = layers.ReLU(negative_slope=relu_slope)(con) if relu else con

    return con


def res_block(x, n_group, bottleneck, down_sample=False, normalizations=(None, None, None)):
    f = int(x.shape[-1])

    con = con_block(x, n_group * bottleneck, kernel=1, normalization=normalizations[0])
    con = group_block(con, n_group, bottleneck, stride=2 if down_sample else 1, normalization=normalizations[1])
    con = con_block(con, f, kernel=1, relu=down_sample, normalization=normalizations[-1])

    x = pooling_layer()(x) if down_sample else x
    res = layers.concatenate([x, con]) if down_sample else layers.add([x, con])
    res = res if down_sample else layers.ReLU(negative_slope=relu_slope)(res)

    return res


def res_block_routing(x, n_group, bottleneck, down_sample=False, normalizations=(None, None, None)):
    f = int(x.shape[-1])

    con = con_block(x, n_group * bottleneck, kernel=1, normalization=normalizations[0])
    con = group_block(con, n_group, bottleneck, stride=2 if down_sample else 1, normalization=normalizations[1])
    con = routing_block(con, n_group, f, kernel=1, relu=down_sample, normalization=normalizations[-1])

    x = pooling_layer()(x) if down_sample else x
    res = layers.concatenate([x, con]) if down_sample else layers.add([x, con])
    res = res if down_sample else layers.ReLU(negative_slope=relu_slope)(res)

    return res
