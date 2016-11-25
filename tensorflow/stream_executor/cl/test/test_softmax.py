from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest


def softmax(x):
    batch_size = x.shape[0]
    e_x = np.exp(x - np.max(x, 1).reshape(batch_size, -1))
    return e_x / e_x.sum(1).reshape(batch_size, -1)


@pytest.mark.parametrize('size', [
    (1, 3),
    (2, 3),
    (3, 10)
])
def test_softmax(size):
    print('size', size)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [None, None], 'a')
            tf_out = tf.nn.softmax(tf_a, name="out")

            np.random.seed(123)
            # shape = (3, 10)
            # shape = (3, 10)
            a = np.random.uniform(10, size=size)
            expected = softmax(a)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                gpu_out = sess.run(tf_out, {tf_a: a})
            print('a', a)
            print('expected', expected)
            print('gpu', gpu_out)
            diff = np.abs(gpu_out - expected).max()
            print('diff', diff)
            assert diff <= 1e-4
