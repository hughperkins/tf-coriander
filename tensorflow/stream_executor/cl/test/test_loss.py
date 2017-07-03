from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest
import sys
from tensorflow.python.ops import array_ops


rows = 10
cols = 15


def test_cross_entropy():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_pred = tf.placeholder(tf.float32, [rows, cols], 'pred')
            tf_y = tf.placeholder(tf.float32, [rows, cols], 'y')
            # tf_b = tf.placeholder(tf.float32, [rows, cols], 'a')
            # tf_out = tf.concat(0, [tf_a, tf_b])
            # tf_out1 = tf.concat(1, [tf_a, tf_b])
            tf_out = tf.nn.softmax_cross_entropy_with_logits(logits=tf_pred, labels=tf_y)

            pred = np.random.randn(rows, cols).astype(np.float32)
            y = np.random.randn(rows, cols).astype(np.float32)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                out = sess.run(tf_out, {tf_pred: pred, tf_y: y})
            # print('a', a)
            print('out', out)
            # print('out1', out1)
            # diff = np.abs(gpu_out - expected).max()
            # print('diff', diff)
            # assert diff <= 1e-4
