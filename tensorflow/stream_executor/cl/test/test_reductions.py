from __future__ import print_function
import tensorflow as tf
import numpy as np


def test(tf_func, py_func):
    print('func', tf_func)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [None, None], 'a')
            # tf_b = tf.placeholder(tf.float32, [None, None], 'b')
            tf_c = tf.__dict__[tf_func](tf_a, name="c")

        np.random.seed(123)
        shape = (1, 10)
        a = np.random.choice(50, shape) / 25
        # b = np.random.choice(50, shape) / 25

        ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
        print('ar', ar)
        # print('br', br)
        print('cr', cr)
        c_py = eval(py_func)
        diff = np.abs(c_py - cr).max()
        print('diff', diff)
        assert diff < 1e-4, 'failed for %s' % tf_func


funcs = {
    'reduce_sum': 'np.sum(a)',
    'reduce_max': 'np.max(a)',
    # 'reduce_min': 'np.min(a)',
    'reduce_prod': 'np.prod(a)',
    'reduce_mean': 'np.mean(a)'
}
for tf_func, py_func in funcs.items():
    test(tf_func, py_func)
