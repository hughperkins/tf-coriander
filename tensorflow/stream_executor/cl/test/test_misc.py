from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest


rows = 10
cols = 15


def test_indexing():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [rows, cols], 'a')
            tf_out = tf_a[0]
            tf_out2 = tf_a[0:3]
            tf_out3 = tf.transpose(tf_a)[0:3]

            a = np.random.randn(rows, cols).astype(np.float32)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                out, out2, out3 = sess.run((tf_out, tf_out2, tf_out3), {tf_a: a})
            print('a', a)
            print('out', out)
            print('out2', out2)
            print('out3', out3)
            # diff = np.abs(gpu_out - expected).max()
            # print('diff', diff)
            # assert diff <= 1e-4


def test_slice():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [rows, cols], 'a')
            tf_out = tf.slice(tf_a, [1, 0], [2, 0])

            a = np.random.randn(rows, cols).astype(np.float32)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                out = sess.run(tf_out, {tf_a: a})
            print('a', a)
            print('out', out)
            # diff = np.abs(gpu_out - expected).max()
            # print('diff', diff)
            # assert diff <= 1e-4


def test_strided_slice():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [rows, cols], 'a')
            tf_out = tf.strided_slice(tf_a, [1, 0], [2, 0], [2, 1])

            a = np.random.randn(rows, cols).astype(np.float32)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                out = sess.run(tf_out, {tf_a: a})
            print('a', a)
            print('out', out)
            # diff = np.abs(gpu_out - expected).max()
            # print('diff', diff)
            # assert diff <= 1e-4


def test_concat():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [rows, cols], 'a')
            tf_b = tf.placeholder(tf.float32, [rows, cols], 'a')
            tf_out = tf.concat(0, [tf_a, tf_b])
            tf_out1 = tf.concat(1, [tf_a, tf_b])

            a = np.random.randn(rows, cols).astype(np.float32)
            b = np.random.randn(rows, cols).astype(np.float32)
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                out, out1 = sess.run((tf_out, tf_out1), {tf_a: a, tf_b: b})
            print('a', a)
            print('out', out)
            print('out1', out1)
            # diff = np.abs(gpu_out - expected).max()
            # print('diff', diff)
            # assert diff <= 1e-4
