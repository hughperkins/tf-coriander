from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest


funcs = {
    'tanh': 'np.tanh(a)',
    'neg': 'np.negative(a)',
    'exp': 'np.exp(a)',
    'sigmoid': '1/(1+np.exp(-a))',
    'sqrt': 'np.sqrt(a)',
    'log': 'np.log(a)',
    'abs': 'np.abs(a)',
    'floor': 'np.floor(a)',
    'ceil': 'np.ceil(a)',
    'square': 'np.square(a)',
    'argmax': 'np.argmax(a, 1)',
    'argmin': 'np.argmin(a, 1)',
}


def nop(val):
    return val


def get_test_params():
    tests = []
    for dtype in ['uint8', 'float32', 'int32']:
        for tf_func, py_func in funcs.items():
            if 'int' in dtype and tf_func in [
                    'ceil', 'floor', 'exp', 'sigmoid', 'log', 'sqrt', 'tanh', 'argmax', 'argmin']:
                continue
            if dtype == 'uint8' and tf_func in ['abs', 'square', 'neg']:
                continue
            mark = nop
            tests.append({'mark': mark, 'dtype': dtype, 'tf_func': tf_func, 'py_func': py_func})
    return tests


@pytest.mark.parametrize(
    'dtype, tf_func, py_func',
    [d['mark']((d['dtype'], d['tf_func'], d['py_func'])) for d in get_test_params()])
def test(tf_func, py_func, dtype):
    print('func', tf_func, dtype)
    np_dtype = eval('np.%s' % dtype)
    tf_dtype = eval('tf.%s' % dtype)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf_dtype, [None, None], 'a')
            if tf_func in ['argmax', 'argmin']:
                tf_c = tf.__dict__[tf_func](tf_a, 1, name="c")
            else:
                tf_c = tf.__dict__[tf_func](tf_a, name="c")
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

                np.random.seed(123)
                shape = (3, 10)
                a = np.random.choice(50, shape) / 50
                if 'sqrt' not in tf_func and 'log' not in tf_func:
                    a -= 0.5
                if 'int' in dtype:
                    a *= 10
                a = a.astype(np_dtype)

                ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
                print('original ', ar)
                c_py = eval(py_func)
                diff = np.abs(c_py - cr).max()
                print('expected ', c_py)
                print('gpu ', cr)
                print('diff', diff)
                assert diff < 1e-4, 'failed for %s' % tf_func
