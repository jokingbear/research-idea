import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32, shape=[None, 28, 28, 128])
w = tf.placeholder(tf.float32, shape=[3, 3, 4, 128])

arr_w = []
for i in range(32):
    w_i = w[..., i * 4: (i + 1) * 4]
    w_i = tf.pad(w_i, [(0, 0), (0, 0), (i * 4, (32 - i - 1) * 4), (0, 0)])

    arr_w.append(w_i)

ws = tf.concat(arr_w, -1)

cnn = tf.nn.conv2d(a, ws, [1, 2, 2, 1], "SAME")
cnn1 = tf.nn.conv2d(a[..., 4:8], w[..., 4:8], [1, 2, 2, 1], "SAME")

with tf.Session() as sess:
    a_val = np.random.rand(1, 28, 28, 128)
    w_val = np.random.rand(3, 3, 4, 128)

    cnn_val, cnn1_val = sess.run([cnn, cnn1], {a: a_val, w: w_val})

print((cnn_val[..., 4:8] - cnn1_val).sum())


