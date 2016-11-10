from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest


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


def get_test_funcs():
    tests = []
    for tf_func, py_func in funcs.items():
        tests.append({'tf_func': tf_func, 'py_func': py_func})
    return tests


@pytest.mark.parametrize('tf_func,py_func', [(d['tf_func'], d['py_func']) for d in get_test_funcs()])
def test(tf_func, py_func):
    print('func', tf_func)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            with tf.device('/gpu:0'):
                tf_a = tf.placeholder(tf.float32, [None, None], 'a')
                tf_b = tf.placeholder(tf.float32, [None, None], 'b')
                tf_c = tf.__dict__[tf_func](tf_a, tf_b, name="c")

            np.random.seed(123)
            shape = (1, 10)
            a = np.random.choice(50, shape) / 25
            b = np.random.choice(50, shape) / 25

            ar, br, cr = sess.run((tf_a, tf_b, tf_c), {tf_a: a, tf_b: b})
            print('ar', ar)
            print('br', br)
            print('cr', cr)
            c_py = eval(py_func)
            diff = np.abs(c_py - cr).max()
            print('diff', diff)
            assert diff < 1e-4, 'failed for %s' % tf_func
