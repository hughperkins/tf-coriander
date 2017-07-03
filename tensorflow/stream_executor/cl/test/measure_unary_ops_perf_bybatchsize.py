from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import json
import argparse
import test_common

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


def test(tf_func, py_func, dtype, average_over, shape0, force_gpu):
    print('func', tf_func, dtype)
    np_dtype = eval('np.%s' % dtype)
    tf_dtype = eval('tf.%s' % dtype)
    res = None
    with tf.Graph().as_default():
        device_args = '/gpu:0' if force_gpu else ''
        with tf.device(device_args):
            tf_a = tf.placeholder(tf_dtype, [None, None], 'a')
            if tf_func in ['argmax', 'argmin']:
                tf_c = tf.__dict__[tf_func](tf_a, 1, name="c")
            else:
                tf_c = tf.__dict__[tf_func](tf_a, name="c")
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

                np.random.seed(123)
                shape = (shape0, 1600)
                for it in range(1 + average_over):
                    a = np.random.choice(50, shape) / 50
                    if 'sqrt' not in tf_func and 'log' not in tf_func:
                        a -= 0.5
                    else:
                        a += 0.01
                    if 'int' in dtype:
                        a *= 10
                    a = a.astype(np_dtype)

                    # start = time.time()
                    ar, cr = sess.run((tf_a, tf_c), {tf_a: a})
                    probe = ar
                    while isinstance(probe, np.ndarray):
                        probe = probe[0]
                    # time_taken = time.time() - start
                    print(probe)
                    if it == 0:
                        start = time.time()

                # if it == 1:
                time_taken = (time.time() - start) / average_over
                print('time for', tf_func, 'dtype', dtype, 'shape0', shape0, time_taken)
                mb = shape0 * 1600 * 4 / 1024 / 1024
                res = {
                    'tf_func': tf_func, 'shape0': shape0, 'dtype': dtype, 'time': time_taken,
                    'is_cuda': test_common.is_cuda(), 'MB': mb}

                # print('original ', ar)
                c_py = eval(py_func)
                diff = np.abs(c_py - cr).max()
                # print('expected ', c_py)
                # print('gpu ', cr)
                print('diff', diff)
                assert diff < 1e-4, 'failed for %s' % tf_func
    return res


def run(force_gpu, max_size, func_list, dtype_list, average_over):
    results = []
    for dtype in dtype_list:
        for tf_func in func_list:
            py_func = funcs[tf_func]
            if 'int' in dtype and tf_func in [
                    'ceil', 'floor', 'exp', 'sigmoid', 'log', 'sqrt', 'tanh', 'argmax', 'argmin']:
                continue
            if dtype == 'uint8' and tf_func in ['abs', 'square', 'neg']:
                continue
            tens = 1
            while tens <= max_size:
                for units in [1, 3]:
                    shape0 = tens * units
                    if shape0 > max_size:
                        break
                    res = test(tf_func=tf_func, average_over=average_over, py_func=py_func, dtype=dtype, shape0=shape0, force_gpu=force_gpu)
                    results.append(res)
                tens *= 10
    print(json.dumps(results, indent=2))
    test_common.print_as_csv(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-force-gpu', action='store_true')
    parser.add_argument('--dtype-list', default='float32', help='eg: uint8,int32,float32')
    parser.add_argument('--func-list', default='tanh')
    parser.add_argument('--average-over', default=1, type=int, help='number of runs to average over')
    parser.add_argument('--max-size', type=int, default=100000, help='should be power of 10')
    args = parser.parse_args()
    args_dict = args.__dict__
    args_dict['force_gpu'] = not args.no_force_gpu
    del args_dict['no_force_gpu']
    args_dict['func_list'] = args_dict['func_list'].split(',')
    args_dict['dtype_list'] = args_dict['dtype_list'].split(',')
    run(**args_dict)
