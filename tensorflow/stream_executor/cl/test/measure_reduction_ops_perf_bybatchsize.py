from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import json
import argparse
import test_common


show_placement = False


funcs = {
    'reduce_sum': 'np.sum(a, axes)',
    'reduce_max': 'np.max(a, axes)',
    'reduce_min': 'np.min(a, axes)',
    'reduce_prod': 'np.prod(a, axes)',
    'reduce_mean': 'np.mean(a, axes)'
}


def test(tf_func, py_func, axes, force_gpu, shape0):
    print('func', tf_func, 'axes', axes)
    res = None
    with tf.Graph().as_default():
        device_args = '/gpu:0' if force_gpu else ''
        with tf.device(device_args):
            tf_a = tf.placeholder(tf.float32, [None, None], 'a')
            tf_c = tf.__dict__[tf_func](tf_a, reduction_indices=axes, name="c")

            with tf.Session(config=tf.ConfigProto(log_device_placement=show_placement)) as sess:
                np.random.seed(123)
                shape = (shape0, 670)

                for it in range(2):
                    a = np.random.choice(50, shape) / 25 - 1

                    start = time.time()
                    ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
                    probe = ar
                    while isinstance(probe, np.ndarray):
                        probe = probe[0]
                    time_taken = time.time() - start
                    print(probe)
                    if it == 1:
                        print('time for', tf_func, 'axes', axes, time_taken)
                        mb = shape0 * 670 * 4 / 1024 / 1024
                        res = {
                            'tf_func': tf_func, 'shape0': shape0, 'axes': axes, 'time': time_taken, 'MB': mb,
                            'is_cuda': test_common.is_cuda()}
                c_py = eval(py_func)
                diff = np.abs(c_py - cr).max()
                inside_range = diff < 1e-4 * shape[0] * shape[1]
                if not inside_range:
                    print('ar', ar)
                    print('c_py', c_py)
                    print('cr', cr)
                    assert inside_range, 'failed for %s' % tf_func
    return res


def run(force_gpu, max_size, func_list, axis_list):
    results = []
    for axis in axis_list:
        axes = {
            '0': (0,),
            '1': (1,),
            'None': None
        }[axis]
        for tf_func in func_list:
            py_func = funcs[tf_func]
            tens = 1
            while tens <= max_size:
                for units in [1, 3]:
                    shape0 = tens * units
                    if shape0 > max_size:
                        break
                    res = test(tf_func=tf_func, py_func=py_func, axes=axes, shape0=shape0, force_gpu=force_gpu)
                    results.append(res)
                tens *= 10
    print(json.dumps(results, indent=2))
    test_common.print_as_csv(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-force-gpu', action='store_true')
    parser.add_argument('--axis-list', default='0', help='eg: 0,1,None')
    parser.add_argument('--func-list', default='reduce_sum')
    parser.add_argument('--max-size', type=int, default=100000, help='should be power of 10')
    args = parser.parse_args()
    args_dict = args.__dict__
    args_dict['force_gpu'] = not args.no_force_gpu
    del args_dict['no_force_gpu']
    args_dict['func_list'] = args_dict['func_list'].split(',')
    args_dict['axis_list'] = args_dict['axis_list'].split(',')
    run(**args_dict)
