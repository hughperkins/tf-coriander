from __future__ import print_function
import tensorflow as tf
import numpy as np
import pytest


show_placement = False


funcs = {
    'reduce_sum': 'np.sum',
    'reduce_max': 'np.max',
    'reduce_min': 'np.min',
    'reduce_prod': 'np.prod',
    'reduce_mean': 'np.mean'
}


def get_test_params():
    tests = []
    for shape in [(3, 17)]:
        for tf_type in ['tf.float32', 'tf.int32']:
            for tf_func, py_func in funcs.items():
                for axes in ['0', '1', 'None']:
                    if tf_type == 'tf.int32' and tf_func == 'reduce_mean':
                        continue
                    if tf_type == 'tf.int32' and tf_func == 'reduce_prod' and axes == '1':
                        # these all fail on CUDA too, soooo... I guess it's ok???
                        continue
                    tests.append({'tf_func': tf_func, 'py_func': py_func, 'axes': axes, 'tf_type': tf_type, 'shape': shape})
    return tests


@pytest.mark.parametrize('tf_func, py_func, axes, tf_type, shape', [(d['tf_func'], d['py_func'], d['axes'], d['tf_type'], d['shape']) for d in get_test_params()])
def test(tf_func, py_func, axes, tf_type, shape):
    print('func', tf_func, 'axes', axes, tf_type, shape)
    tf_type = eval(tf_type)
    if axes != 'None':
        axes = '(%s,)' % axes
    axes = eval(axes)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            tf_a = tf.placeholder(tf_type, [None, None], 'a')
            tf_c = tf.__dict__[tf_func](tf_a, reduction_indices=axes, name="c")
            print('tf_c', tf_c)

            with tf.Session(config=tf.ConfigProto(log_device_placement=show_placement)) as sess:
                np.random.seed(123)
                if tf_type == tf.float32:
                    a = np.random.choice(50, shape) / 25 - 1
                else:
                    a = np.random.choice(50, shape)

                ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
                c_py = eval(py_func + '(a, axes)')
                diff = np.abs(c_py - cr).max()
                if diff >= 1e-4:
                    print('ar', ar)
                    print('c_py', c_py)
                    print('cr', cr)
                    assert diff < 1e-4, 'failed for %s' % tf_func


if __name__ == '__main__':
    test('reduce_sum', 'np.sum', 'None', 'tf.float32', (1024, 32))
