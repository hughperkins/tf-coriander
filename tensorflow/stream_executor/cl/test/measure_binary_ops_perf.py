from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest
import time


funcs = {
    'add': 'a + b',
    'sub': 'a - b',
    'div': 'a / b',
    'mul': 'a * b',
    'minimum': 'np.minimum(a,b)',
    'maximum': 'np.maximum(a,b)',
    'pow': 'np.power(a,b)',
    'squared_difference': '(a - b) * (a - b)',
    'not_equal': 'np.not_equal(a, b)'
}


def nop(values):
    return values


def get_test_funcs():
    tests = []
    for dtype in ['uint8', 'float32', 'int32']:
        for tf_func, py_func in sorted(funcs.items()):
            if 'int' in dtype and tf_func in ['not_equal', 'pow']:
                continue
            if dtype == 'uint8' and tf_func in [
                    'maximum', 'minimum', 'squared_difference', 'sub', 'add']:
                continue
            mark = nop
            if dtype == 'uint8' and tf_func in ['mul', 'div']:
                mark = pytest.mark.xfail
            test = {'mark': mark, 'dtype': dtype, 'tf_func': tf_func, 'py_func': py_func}

            tests.append(test)
    return tests


@pytest.mark.parametrize(
    'dtype,tf_func,py_func',
    [d['mark']((d['dtype'], d['tf_func'], d['py_func'])) for d in get_test_funcs()])
def test(dtype, tf_func, py_func):
    print('func', tf_func, dtype)
    np_dtype = eval('np.%s' % dtype)
    tf_dtype = eval('tf.%s' % dtype)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            with tf.device('/gpu:0'):
                tf_a = tf.placeholder(tf_dtype, [None, None], 'a')
                tf_b = tf.placeholder(tf_dtype, [None, None], 'b')
                tf_c = tf.__dict__[tf_func](tf_a, tf_b, name="c")

            np.random.seed(123)
            shape = (50000, 1600)
            a = np.random.choice(50, shape) / 25
            b = np.random.choice(50, shape) / 25
            if 'int' in dtype:
                a *= 10
                b *= 10 + 1
            else:
                b += 0.01
            a = a.astype(np_dtype)
            b = b.astype(np_dtype)
            average_over = 10
            for it in range(1 + average_over):
                # start = time.time()
                ar, br, cr = sess.run((tf_a, tf_b, tf_c), {tf_a: a, tf_b: b})
                probe = ar
                while isinstance(probe, np.ndarray):
                    probe = probe[0]
                print(probe)
                if it == 0:
                    start = time.time()
            time_taken = (time.time() - start) / average_over
            print('time for', tf_func, 'dtype', dtype, time_taken)

            # print('a', ar)
            # print('b', br)
            c_py = eval(py_func).astype(np_dtype)
            # print('expected', c_py)
            # print('gpu', cr)
            diff = np.abs(c_py - cr).max()
            # print('diff', diff)
            assert diff < 1e-4, 'failed for %s' % tf_func
