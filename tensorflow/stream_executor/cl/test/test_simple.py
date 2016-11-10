from __future__ import print_function

import tensorflow as tf
import numpy as np


def test_simple():
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            with tf.device('/gpu:0'):
                a = tf.constant([1, 3, 5, 2, 4, 7], dtype=tf.float32, shape=[2, 3], name='a')
                b = tf.constant([3, 4, 4, 6, 6, 5], dtype=tf.float32, shape=[2, 3], name='b')
                c = tf.add(a, b, name="c")
                print(sess.run(c))
