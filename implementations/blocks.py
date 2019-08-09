from keras import layers, Sequential
from implementations import group_conv as gc

relu_slope = 0.2


def con_block(name, f, kernel=3, stride=1, relu=True, normalization=None):
    block = Sequential(name=name)

    block.add(layers.Conv2D(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal"))
    block.add(normalization) if normalization else None
    block.add(layers.ReLU(negative_slope=relu_slope)) if relu else None

    return block


def decon_block(name, f, kernel=3, stride=2, relu=True, normalization=None):
    block = Sequential(name=name)

    block.add(layers.Conv2DTranspose(f, kernel, strides=stride, padding="same", kernel_initializer="he_normal"))
    block.add(normalization) if normalization else None
    block.add(layers.ReLU(negative_slope=relu_slope)) if relu else None

    return block


def group_block(name, n_group, f, kernel=3, stride=1, relu=True, normalization=None):
    block = Sequential(name=name)

    block.add(gc.GroupConv2D(n_group, f, kernel, strides=stride, kernel_initializer="he_normal"))
    block.add(normalization) if normalization else None
    block.add(layers.ReLU(negative_slope=0.2)) if relu else None

    return block


def routing_block(name, n_group, f, kernel=3, stride=1, relu=True, normalization=None, n_iter=3):
    block = Sequential(name=name)

    block.add(gc.GroupRouting2D(n_group, f, kernel, strides=stride, n_iter=n_iter))
    block.add(normalization) if normalization else None
    block.add(layers.ReLU(negative_slope=relu_slope)) if relu else None

    return block


def res_block(name, n_group, f, bottleneck, down_sample=False, normalizations=(None, None, None)):
    block = Sequential([
        con_block(f"{name}_embedding", n_group * bottleneck, kernel=1, normalization=normalizations[0]),
        group_block(f"{name}_group", n_group, bottleneck, kernel=3, stride=2 if down_sample else 1,
                    normalization=normalizations[1]),
        con_block(f"{name}_return", f, kernel=1, relu=down_sample, normalization=normalizations[-1])
    ], name=name)

    return lambda x: layers.concatenate([layers.MaxPool2D()(x), block(x)]) if down_sample else \
        layers.ReLU(negative_slope=relu_slope)(layers.add([x, block(x)]))


def res_block_routing(name, n_group, f, bottleneck, down_sample=False, normalizations=(None, None, None)):
    block = Sequential([
        con_block(f"{name}_embedding", n_group * bottleneck, kernel=1, normalization=normalizations[0]),
        group_block(f"{name}_group", n_group, bottleneck, kernel=3, stride=2 if down_sample else 1,
                    normalization=normalizations[1]),
        routing_block(f"{name}_return", n_group, f, kernel=1, relu=down_sample, normalization=normalizations[-1])
    ], name=name)

    return lambda x: layers.concatenate([layers.MaxPool2D()(x), block(x)]) if down_sample else \
        layers.ReLU(negative_slope=relu_slope)(layers.add([x, block(x)]))
