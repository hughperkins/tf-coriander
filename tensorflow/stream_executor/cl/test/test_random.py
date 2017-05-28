from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest
import sys
from tensorflow.python.ops import array_ops


def test_random_normal():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            W_t = tf.Variable(tf.random_normal([3, 4]))
            mu_t = tf.reduce_mean(W_t)
            var_t = tf.reduce_mean(W_t * W_t)

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.initialize_all_variables())
                W, mu, var = sess.run((W_t, mu_t, var_t))
            print('W', W)
            print('mu', mu, 'var', var)
