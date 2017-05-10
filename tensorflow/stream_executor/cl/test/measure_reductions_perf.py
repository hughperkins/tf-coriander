from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import pytest


show_placement = False


funcs = {
    'reduce_sum': 'np.sum(a, axes)',
    'reduce_max': 'np.max(a, axes)',
    'reduce_min': 'np.min(a, axes)',
    'reduce_prod': 'np.prod(a, axes)',
    'reduce_mean': 'np.mean(a, axes)'
}


def get_test_params():
    tests = []
    for tf_func, py_func in sorted(funcs.items()):
        for axes in [(0,), (1,), None]:
            tests.append({'tf_func': tf_func, 'py_func': py_func, 'axes': axes})
    return tests


@pytest.mark.parametrize('tf_func, py_func, axes', [(d['tf_func'], d['py_func'], d['axes']) for d in get_test_params()])
def test(tf_func, py_func, axes):
    print('func', tf_func, 'axes', axes)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf.float32, [None, None], 'a')
            tf_c = tf.__dict__[tf_func](tf_a, reduction_indices=axes, name="c")

            with tf.Session(config=tf.ConfigProto(log_device_placement=show_placement)) as sess:
                np.random.seed(123)
                shape = (250000, 670)

                for it in range(2):
                    a = np.random.choice(50, shape) / 25 - 1

                    start = time.time()
                    ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
                    probe = ar
                    while isinstance(probe, np.ndarray):
                        probe = probe[0]
                    print(probe)
                    if it == 1:
                        print('time for', tf_func, 'axes', axes, time.time() - start)
                c_py = eval(py_func)
                diff = np.abs(c_py - cr).max()
                inside_range = diff < 1e-4 * shape[0] * shape[1]
                if not inside_range:
                    print('ar', ar)
                    print('c_py', c_py)
                    print('cr', cr)
                    assert inside_range, 'failed for %s' % tf_func
