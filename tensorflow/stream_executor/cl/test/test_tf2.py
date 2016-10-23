from __future__ import print_function

import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    tf_a = tf.placeholder(tf.float32, [None, None], 'a')
    tf_b = tf.placeholder(tf.float32, [None, None], 'b')
    tf_c = tf.add(tf_a, tf_b, name="c")

    np.random.seed(123)
    shape = (5, 13)
    a = np.random.choice(50, shape)
    b = np.random.choice(50, shape)

    ar, br, cr = sess.run((tf_a, tf_b, tf_c), {tf_a: a, tf_b: b})
    print('ar', ar)
    print('br', br)
    print('cr', cr)
    print('diff', cr - a - b)
