from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse


def run_op(op):
    graph = tf.Graph()
    with graph.as_default():
        a = tf.constant([1, 3, 5, 2, 4, 7], dtype=tf.float32, shape=[2, 3], name='a')
        # b = tf.constant([3, 4, 4, 6, 6, 5], dtype=tf.float32, shape=[2, 3], name='b')
        c = eval('tf.%s(a, name="c")' % op)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        with sess.as_default():
            with tf.device('/gpu:0'):
                print('running sess.run a')
                try:
                    # print(sess.run(a))
                    print(sess.run(c))
                except Exception as e:
                    print('exception')
                    print(e)
                print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, default='tanh')
    args = parser.parse_args()
    run_op(**args.__dict__)
