from __future__ import print_function
import tensorflow as tf
import numpy as np


def test(tf_func, py_func):
    print('func', tf_func)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [None, None], 'a')
            tf_c = tf.__dict__[tf_func](tf_a, name="c")

        np.random.seed(123)
        shape = (1, 10)
        a = np.random.choice(50, shape) / 50
        if 'sqrt' not in tf_func and 'log' not in tf_func:
            a -= 0.5

        ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
        print('original ', ar)
        c_py = eval(py_func)
        diff = np.abs(c_py - cr).max()
        print('expected ', c_py)
        print('gpu ', cr)
        print('diff', diff)
        assert diff < 1e-4, 'failed for %s' % tf_func


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
    'square': 'np.square(a)'
}
for tf_func, py_func in funcs.items():
    test(tf_func, py_func)
