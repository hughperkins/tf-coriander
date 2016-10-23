from __future__ import print_function
import tensorflow as tf
import numpy as np


def test(tf_func, py_func):
    print('func', tf_func)
    with tf.Session() as sess:
        tf_a = tf.placeholder(tf.float32, [None, None], 'a')
        tf_c = tf.__dict__[tf_func](tf_a, name="c")

        np.random.seed(123)
        shape = (1, 10)
        a = np.random.choice(50, shape) / 25

        ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
        print('ar', ar)
        print('cr', cr)
        c_py = eval(py_func)
        diff = np.abs(c_py - cr).max()
        print('diff', diff)
        assert diff < 1e-4, 'failed for %s' % tf_func


funcs = {
    'tanh': 'np.tanh(a)'
}
for tf_func, py_func in funcs.items():
    test(tf_func, py_func)
