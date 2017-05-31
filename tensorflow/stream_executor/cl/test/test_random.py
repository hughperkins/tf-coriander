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


def test_random_func(func_name, shape):
    print('func_name', func_name)
    func = eval(func_name)
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            W_t = tf.Variable(func(shape, seed=123))

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.initialize_all_variables())
                W_cpu = sess.run(W_t)
        with tf.device('/gpu:0'):
            W_t = tf.Variable(func(shape, seed=123))

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.initialize_all_variables())
                W_gpu = sess.run(W_t)
            if np.prod(W_gpu.shape) < 20:
                print('W_cpu', W_cpu)
                print('W_gpu', W_gpu)
            else:
                print('W_cpu.reshape(-1)[:20]', W_cpu.reshape(-1)[:20])
                print('W_gpu.reshape(-1)[:20]', W_gpu.reshape(-1)[:20])
            assert np.all(W_cpu == W_gpu)


@pytest.mark.parametrize(
    'shape',
    shapes)
def test_random_normal(shape):
    test_random_func('tf.random_normal', shape)


@pytest.mark.parametrize(
    'shape',
    shapes)
def test_random_uniform(shape):
    test_random_func('tf.random_uniform', shape)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please run using py.test')
    else:
        eval('%s((3, 4))' % sys.argv[1])
