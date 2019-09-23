from keras import layers
from tf_modules import custom_layers as clayers

con_layer = layers.Conv2D
decon_layer = layers.Conv2DTranspose
group_layer = clayers.GroupConv2D
routing_layer = clayers.DynamicRouting2D
pooling_layer = layers.AveragePooling2D


def default_normalization(**kwargs):
    return layers.BatchNormalization()


def default_activation(**kwargs):
    return layers.LeakyReLU(alpha=0.2)


def dense_block(x, f, normalization=None, dropout=None, activation=default_activation):
    normalization = normalization or default_normalization
    dropout = layers.Dropout(dropout) if dropout is not None else (lambda arg: arg)
    activation = activation or (lambda arg: arg)

    d = layers.Dense(f, use_bias=False, kernel_initializer="he_normal")(x)
    d = normalization()(d)
    d = dropout(d)
    d = activation()(d)

    return d


def con_block(x, f, n_group=1, kernel=3, stride=1, dilation=1, normalization=None, activation=default_activation):
    normalization = normalization or default_normalization
    activation = activation() if activation else (lambda arg: arg)

    if n_group == 1:
        con = con_layer(f, kernel, strides=stride, padding="same", dilation_rate=dilation,
                        use_bias=False, kernel_initializer="he_normal")(x)
    else:
        con = group_layer(n_group, f, kernel, stride, dilation, use_bias=False)(x)

    con = normalization(n_group=n_group)(con)
    con = activation(con)

    return con


def decon_block(x, f, kernel=3, stride=2, dilation=1, normalization=None, activation=default_activation):
    normalization = normalization or default_normalization
    activation = activation() if activation else (lambda arg: arg)

    con = decon_layer(f, kernel, strides=stride, padding="same", dilation_rate=dilation,
                      use_bias=False, kernel_initializer="he_normal")(x)
    con = normalization()(con)
    con = activation(con)

    return con


def routing_block(x, n_group, f, n_iter, normalization=None, activation=default_activation):
    if n_iter == 1:
        return con_block(x, f, kernel=1, normalization=normalization, activation=activation)

    normalization = normalization or default_normalization
    activation = activation() if activation else (lambda arg: arg)

    con = routing_layer(n_group, f, n_iter, kernel_initializer="he_normal")(x)
    con = normalization()(con)
    con = activation(con)

    return con


def res_block(x, n_group, bottleneck, n_iter=3, down_sample=False, normalization=None, activation=default_activation):
    f = int(x.shape[-1])
    b = bottleneck
    merge_layer = layers.concatenate if down_sample else layers.add

    con = con_block(x, n_group * b, kernel=1, normalization=normalization, activation=activation)
    con = con_block(con, b, n_group, stride=2 if down_sample else 1, normalization=normalization, activation=activation)
    con = routing_block(con, n_group, f, n_iter, normalization=normalization,
                        activation=activation if down_sample else None)

    activation = activation() if not down_sample and activation else (lambda arg: arg)
    x = pooling_layer()(x) if down_sample else x
    merge = merge_layer([x, con])
    activate = activation(merge)

    return activate
