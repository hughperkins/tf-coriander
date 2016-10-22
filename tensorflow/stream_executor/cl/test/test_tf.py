from __future__ import print_function

import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a = tf.constant([1, 3, 5, 2, 4, 7], dtype=tf.float32, shape=[2, 3], name='a')
    b = tf.constant([3, 4, 4, 6, 6, 5], dtype=tf.float32, shape=[2, 3], name='b')
    c = tf.add(a, b, name="c")
    print(sess.run(c))
