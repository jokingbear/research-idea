from tensorflow.keras import layers
from implementations import group_conv as gc


con_layer = layers.Conv2D
decon_layer = layers.Conv2DTranspose
group_layer = gc.GroupConv2D
routing_layer = gc.DynamicRouting
pooling_layer = layers.AveragePooling2D


def activation_layer(x): return layers.ReLU(negative_slope=0.2)(x)


def con_block(x, f, kernel=3, stride=1, relu=True, normalization=None):
    con = con_layer(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
    con = normalization(con) if normalization else con
    con = activation_layer(con) if relu else con

    return con


def decon_block(x, f, kernel=3, stride=2, relu=True, normalization=None):
    decon = decon_layer(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal", use_bias=False)(x)
    decon = normalization(decon) if normalization else decon
    decon = activation_layer(decon) if relu else decon

    return decon


def group_block(x, n_group, f, kernel=3, stride=1, dilation=1, relu=True, normalization=None):
    con = group_layer(n_group, f, kernel, strides=stride, dilations=dilation, kernel_initializer="he_normal",
                      use_bias=False)(x)
    con = normalization(con) if normalization else con
    con = activation_layer(con) if relu else con

    return con


def routing_block(x, n_group, f, relu=True, normalization=None, n_iter=3):
    if n_iter == 1:
        return con_block(x, f, kernel=1, relu=relu, normalization=normalization)

    con = routing_layer(n_group, f, n_iter=n_iter, use_bias=False)(x)
    con = normalization(con) if normalization else con
    con = activation_layer(con) if relu else con

    return con


def res_block(x, n_group, bottleneck, n_iter=3, down_sample=False, normalizations=(None, None, None)):
    f = int(x.shape[-1])

    con = con_block(x, n_group * bottleneck, kernel=1, normalization=normalizations[0])
    con = group_block(con, n_group, bottleneck, stride=2 if down_sample else 1, normalization=normalizations[1])
    con = routing_block(con, n_group, f, relu=down_sample, normalization=normalizations[-1], n_iter=n_iter)

    x = pooling_layer()(x) if down_sample else x
    res = layers.concatenate([x, con]) if down_sample else layers.add([x, con])
    res = res if down_sample else activation_layer(res)

    return res


def res_scale_block(x, n_group, bottleneck, n_iter=3, normalizations=(None, None, None)):
    f = x.shape[-1]
    sg = n_group // 4
    b = bottleneck

    cons = [con_block(x, sg * b, kernel=1, normalization=normalizations[0]()) for i in range(4)]
    cons = [group_block(c, sg, b, stride=2, normalization=normalizations[1](), dilation=6*i + 1)
            for i, c in zip(range(4), cons)]
    cons = [routing_block(c, sg, f // 4, normalization=normalizations[-1](), n_iter=n_iter) for c in cons]

    x = pooling_layer()(x)

    return layers.concatenate([x, *cons])
