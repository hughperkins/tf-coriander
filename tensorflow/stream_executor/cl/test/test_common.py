import tensorflow as tf


def print_as_csv(results):
    columns = sorted(results[0].keys())
    print('\t'.join(columns))
    for result in results:
        line_list = []
        for column in columns:
            line_list.append(str(result[column]))
        print('\t'.join(line_list))


def is_cuda():
    # this might not always differentiate cuda vs opencl, but it does so for now
    # (using cuda-on-cl implemented opencl)
    return tf.pywrap_tensorflow.IsGoogleCudaEnabled()
