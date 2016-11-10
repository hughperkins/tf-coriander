from __future__ import print_function

import tensorflow as tf
import numpy as np


def test_blas():
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            with tf.device('/gpu:0'):
                tf_a = tf.placeholder(tf.float32, [None, None], 'a')
                tf_b = tf.placeholder(tf.float32, [None, None], 'b')
                tf_matmul = tf.matmul(tf_a, tf_b, name="matmul")

            np.random.seed(123)
            shape = (5, 5)
            a = np.random.choice(50, shape)
            b = np.random.choice(50, shape)

            ar, br, matmulr = sess.run((tf_a, tf_b, tf_matmul), {tf_a: a, tf_b: b})
            print('a', ar)
            print('b', br)
            expected = a.dot(b)
            print('expected', expected)
            print('gpu', matmulr)
            print('diff', matmulr - expected)
            assert np.abs(matmulr - expected).max() <= 1e-4
