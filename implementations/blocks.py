from tensorflow.keras import layers
from implementations import group_conv as gc


con_layer = layers.Conv2D
decon_layer = layers.Conv2DTranspose
group_layer = gc.GroupConv2D
routing_layer = gc.DynamicRouting
pooling_layer = layers.AveragePooling2D


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
    con = group_layer(n_group, f, kernel, strides=stride, dilations=dilation, kernel_initializer="he_normal")(x)
    con = normalization(con) if normalization else con
    con = activation_layer(con) if relu else con

    return con


def routing_block(x, n_group, f, relu=True, normalization=None, n_iter=3):
    con = group_layer(n_group, f, 1, kernel_initializer="he_normal", use_bias=False)(x)
    con = routing_layer(n_group, n_iter=n_iter)(con)
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
        con = routing_block(con, n_group, f, relu=down_sample, normalization=normalizations[-1], n_iter=n_iter)

    x = pooling_layer()(x) if down_sample else x
    res = layers.concatenate([x, con]) if down_sample else layers.add([x, con])
    res = res if down_sample else activation_layer(res)

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
