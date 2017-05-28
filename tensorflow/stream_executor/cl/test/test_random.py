from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest
import sys
from tensorflow.python.ops import array_ops


shapes = [
    (3, 4),
    (50, 70, 12)
]


@pytest.mark.parametrize(
    'shape',
    shapes)
def test_random_normal(shape):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            W_t = tf.Variable(tf.random_normal(shape))
            mu_t = tf.reduce_mean(W_t)
            var_t = tf.reduce_mean(W_t * W_t)

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.initialize_all_variables())
                W, mu, var = sess.run((W_t, mu_t, var_t))
            if np.prod(W.shape) < 20:
                print('W', W)
            else:
                print('W.reshape(-1)[:20]', W.reshape(-1)[:20])
            print('mu', mu, 'var', var)
            assert abs(mu) < 1.0
            assert var > 0.05
            assert var < 4.0


@pytest.mark.parametrize(
    'shape',
    shapes)
def test_random_uniform(shape):
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            W_t = tf.Variable(tf.random_uniform(shape))
            mu_t = tf.reduce_mean(W_t)
            var_t = tf.reduce_mean(W_t * W_t)

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.initialize_all_variables())
                W, mu, var = sess.run((W_t, mu_t, var_t))
            if np.prod(W.shape) < 20:
                print('W', W)
            else:
                print('W.reshape(-1)[:20]', W.reshape(-1)[:20])
            print('mu', mu, 'var', var)
            assert abs(mu) < 1.0
            assert var > 0.05
            assert var < 4.0

# @pytest.mark.parametrize(
#     'dtype, tf_func, py_func',
#     [d['mark']((d['dtype'], d['tf_func'], d['py_func'])) for d in get_test_params()])
