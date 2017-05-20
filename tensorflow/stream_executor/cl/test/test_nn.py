from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest


rows = 10
cols = 15


def test_relu():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [rows, cols], 'a')
            tf_out = tf.nn.relu(tf_a)

            a = np.random.randn(rows, cols).astype(np.float32)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                out = sess.run(tf_out, {tf_a: a})
            print('a', a)
            print('out', out)
            # diff = np.abs(gpu_out - expected).max()
            # print('diff', diff)
            # assert diff <= 1e-4
