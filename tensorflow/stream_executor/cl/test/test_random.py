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

seed = 123


def _test_random_func(func_name, shape):
    print('func_name', func_name)
    func = eval(func_name)
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            W_t = tf.Variable(func(shape, seed=seed))

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.initialize_all_variables())
                W_cpu = sess.run(W_t)
        with tf.device('/gpu:0'):
            W_t = tf.Variable(func(shape, seed=seed))

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.initialize_all_variables())
                W_gpu = sess.run(W_t)
            if np.prod(np.array(shape)) < 20:
                print('W_cpu', W_cpu)
                print('W_gpu', W_gpu)
            else:
                print('W_cpu.reshape(-1)[:20]', W_cpu.reshape(-1)[:20])
                print('W_gpu.reshape(-1)[:20]', W_gpu.reshape(-1)[:20])
            assert np.all(np.abs(W_cpu - W_gpu) < 1e-4)


@pytest.mark.parametrize(
    'shape',
    shapes)
def test_random_normal(shape):
    _test_random_func('tf.random_normal', shape)


@pytest.mark.parametrize(
    'shape',
    shapes)
def test_random_uniform(shape):
    _test_random_func('tf.random_uniform', shape)


@pytest.mark.parametrize(
    'shape',
    shapes)
@pytest.mark.skip(reason='Causes abort currently')
def test_truncated_normal(shape):
    _test_random_func('tf.truncated_normal', shape)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please run using py.test')
    else:
        eval('%s((3, 4))' % sys.argv[1])
