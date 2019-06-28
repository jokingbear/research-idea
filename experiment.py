import tensorflow as tf
import numpy as np

w = tf.placeholder(tf.float32, shape=[3, 3, 4, 64])

img = tf.placeholder(tf.float32, shape=[None, 28, 28, 16])

cnn = tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding="SAME")


with tf.Session() as sess:
    tmp = sess.run(cnn, {w: np.random.rand(3, 3, 4, 64),
                         img: np.random.rand(1, 28, 28, 16)})
