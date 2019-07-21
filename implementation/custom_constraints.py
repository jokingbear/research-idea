import tensorflow as tf

from keras import initializers as inits, backend as K


def spectral_norm_constraint(mode="fan_in",
                             initializer=inits.normal(stddev=1)):

    def spectral_norm(w):
        w1 = w if mode == "fan_in" else tf.transpose(w, [0, 1, 3, 2])

        fo = int(w1.shape[-1])

        w1 = tf.reshape(w1, [-1, fo])

        fi = int(w1.shape[0])

        u = tf.Variable(initializer([1, fi]), trainable=False, name="singular_vector")

        v = tf.nn.l2_normalize(u @ w1)
        u1 = tf.nn.l2_normalize(tf.matmul(v, w1, transpose_b=True))

        u = K.in_train_phase(tf.assign(u, u1), u1)
        sigma = tf.reduce_sum((u @ w1) * v)

        return w / sigma

    return spectral_norm

