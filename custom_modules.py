from keras import layers
from group_conv import GroupConv2D


def con_block(x, n_filter, kernel_size, strides=1, relu=True, normalization=None):
    con = layers.Conv2D(n_filter, kernel_size, strides=strides, padding="same",
                        kernel_initializer="he_normal", kernel_regularizer=None)(x)

    con = normalization(con) if normalization else con
    con = layers.ReLU()(con) if relu else con

    return con


def group_block(x, n_group, n_filter, kernel_size, strides=1, relu=True, normalization=None):
    gcon = GroupConv2D(n_group, n_filter, kernel_size, strides=strides, padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=None)(x)

    gcon = normalization(gcon) if normalization else gcon
    gcon = layers.ReLU()(gcon) if relu else gcon

    return gcon


def res_block(x, k=3, down_sample=False):
    f = int(x.shape[-1])

    con = x
    con = con_block(con, f // 2, 1, normalization=layers.BatchNormalization())
    con = con_block(con, f // 2, k, normalization=layers.BatchNormalization(), strides=2 if down_sample else 1)
    con = con_block(con,      f, 1, normalization=layers.BatchNormalization(), relu=down_sample)

    x = layers.MaxPool2D()(x) if down_sample else x

    res = layers.concatenate([con, x]) if down_sample else layers.add([con, x])
    res = res if down_sample else layers.ReLU()(res)

    return res


def res_block2(x, n_group, bottleneck, k=3, down_sample=False):
    f = int(x.shape[-1])
    
    con = x
    con = con_block(con, n_group * bottleneck, 1, normalization=layers.BatchNormalization())
    con = group_block(con, n_group, bottleneck, k, strides=2 if down_sample else 1,
                      normalization=layers.BatchNormalization())
    con = con_block(con, f, 1, relu=down_sample, normalization=layers.BatchNormalization())

    x = layers.MaxPool2D()(x) if down_sample else x

    res = layers.concatenate([x, con]) if down_sample else layers.add([x, con])
    res = res if down_sample else layers.ReLU()(res)

    return res

