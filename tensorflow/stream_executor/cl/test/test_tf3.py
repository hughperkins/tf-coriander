from __future__ import print_function
import tensorflow as tf
import numpy as np


def test(tf_func, py_func):
    print('func', tf_func)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
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


# for func in ['add', 'div', 'sub', 'max', 'min', 'mul', 'pow', 'squared_difference', 'not_equal_to']:
funcs = {
    'add': 'a + b',
    # 'sub': 'a - b',
    # 'div': 'a / b',
    # 'mul': 'a * b',
    # 'minimum': 'np.minimum(a,b)',
    # 'maximum': 'np.maximum(a,b)',
    # 'pow': 'np.power(a,b)',
    # 'squared_difference': '(a - b) * (a - b)',
    # 'not_equal': 'np.not_equal(a, b)'
}
for tf_func, py_func in funcs.items():
    test(tf_func, py_func)


# def test_tffunc_as_string(tf_func, py_func):
#     print('func', tf_func)
#     with tf.Session() as sess:
#         tf_a = tf.placeholder(tf.float32, [None, None], 'a')
#         tf_b = tf.placeholder(tf.float32, [None, None], 'b')
#         tf_c = eval(tf_func)

#         np.random.seed(123)
#         shape = (1, 10)
#         a = np.random.choice(50, shape) / 25
#         b = np.random.choice(50, shape) / 25

#         ar, br, cr = sess.run((tf_a, tf_b, tf_c), {tf_a: a, tf_b: b})
#         print('ar', ar)
#         print('br', br)
#         print('cr', cr)
#         c_py = eval(py_func)
#         diff = np.abs(c_py - cr).max()
#         print('diff', diff)
#         assert diff < 1e-4, 'failed for %s' % tf_func


# test_tffunc_as_string('tf.not_equal(tf_a, tf_b)', 'np.not_equal(a, b)')
