from __future__ import print_function
import tensorflow as tf
import numpy as np


show_placement = False


def test(tf_func, py_func, axes):
    print('func', tf_func, 'axes', axes)
    with tf.device('/gpu:0'):
        tf_a = tf.placeholder(tf.float32, [None, None], 'a')
        tf_c = tf.__dict__[tf_func](tf_a, reduction_indices=axes, name="c")

    np.random.seed(123)
    shape = (3, 17)
    a = np.random.choice(50, shape) / 25 - 1

    ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
    c_py = eval(py_func)
    diff = np.abs(c_py - cr).max()
    if diff >= 1e-4:
        print('ar', ar)
        print('c_py', c_py)
        print('cr', cr)
        assert diff < 1e-4, 'failed for %s' % tf_func


with tf.Session(config=tf.ConfigProto(log_device_placement=show_placement)) as sess:
    funcs = {
        'reduce_sum': 'np.sum(a, axes)',
        'reduce_max': 'np.max(a, axes)',
        'reduce_min': 'np.min(a, axes)',
        'reduce_prod': 'np.prod(a, axes)',
        'reduce_mean': 'np.mean(a, axes)'
    }
    for tf_func, py_func in funcs.items():
        for axes in [(0,), (1,), None]:
            test(tf_func, py_func, axes)
