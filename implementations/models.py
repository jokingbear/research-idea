from implementations import blocks
from tensorflow.keras import layers, Model
from implementations.normalization_layers import GroupNorm


def routing_encoder(input_shape=(512, 512, 1), f=32, n_block=3, n_group=32, n_iter=3, with_se=False,
                    return_block=False, name="res_encoder"):
    x = layers.Input(input_shape)

    n_norm = 4
    bottleneck = 4

    con0 = blocks.con_block(x, f, normalization=GroupNorm(n_norm))

    con = con0
    cons = [con0]
    for i in range(n_block):
        normalizations = [GroupNorm(n_norm * n_group), GroupNorm(n_norm * n_group), GroupNorm(n_norm)]
        con = blocks.res_block(con, n_group, bottleneck, down_sample=True, normalizations=normalizations)

        con = blocks.con_block(con, 2 * f, normalization=GroupNorm(n_norm))
        con = layers.add([blocks.cse_block(con), blocks.sse_block(con)]) if with_se else con

        normalizations = [GroupNorm(n_norm * n_group), GroupNorm(n_norm * n_group), GroupNorm(n_norm)]
        con = blocks.res_block(con, n_group, bottleneck, n_iter=n_iter, normalizations=normalizations)

        f = 2 * f
        bottleneck = 2 * bottleneck
        cons.append(con)

    return Model(x, cons if return_block else con, name=name)
