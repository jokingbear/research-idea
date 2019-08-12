from implementations import blocks, GroupNorm
from keras import layers, Model


def routing_encoder(input_shape=(512, 512, 1), f=32, n_block=3, n_group=32, return_block=False):
    x = layers.Input(input_shape)

    n_norm = 4
    bottleneck = 4

    con0 = blocks.con_block(x, f, normalization=GroupNorm(n_norm))

    con = con0
    cons = [con0]
    for i in range(n_block):
        normalizations = [GroupNorm(n_norm * n_group), GroupNorm(n_norm * n_group), GroupNorm(n_norm)]
        con = blocks.res_block_routing(con, n_group, bottleneck, down_sample=True, normalizations=normalizations)

        con = blocks.con_block(con, 2 * f, normalization=GroupNorm(n_norm))

        normalizations = [GroupNorm(n_norm * n_group), GroupNorm(n_norm * n_group), GroupNorm(n_norm)]
        con = blocks.res_block_routing(con, n_group, bottleneck, normalizations=normalizations)

        f = 2 * f
        bottleneck = 2 * bottleneck
        cons.append(con)

    return Model(x, cons, name="res_encoder") if return_block else Model(x, con, name="res_encoder")
