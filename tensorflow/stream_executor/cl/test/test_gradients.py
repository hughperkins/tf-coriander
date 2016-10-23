from __future__ import print_function

import tensorflow as tf
import numpy as np

learning_rate = 0.1

# lets learn or
# we'll use one-hot, with 2 binary inputs, so 4 input neurons in total
# output is one binary value, so 2 output neurons (since one-hot)
data = [
    {'input': [False, False], 'output': False},
    {'input': [False, True], 'output': True},
    {'input': [True, False], 'output': True},
    {'input': [True, True], 'output': True}
]
batch_size = len(data)
X = np.zeros((batch_size, 4), dtype=np.float32)
y = np.zeros((batch_size, 2), dtype=np.float32)
for n, ex in enumerate(data):
    input = ex['input']
    output = ex['output']
    if input[0]:
        X[n][1] = 1
    else:
        X[n][0] = 1
    if input[1]:
        X[n][3] = 1
    else:
        X[n][2] = 1
    if output:
        y[n][1] = 1
    else:
        y[n][0] = 1
print('X', X)
print('y', y)

with tf.Session() as sess:
    tf_x = tf.placeholder(tf.float32, [None, 4], 'x')
    tf_W = tf.Variable(tf.zeros([4, 2]), 'W')
    tf_bias = tf.Variable(tf.zeros(2,), 'bias')
    tf_out = tf.matmul(tf_x, tf_W, name="out") + tf_bias

    np.random.seed(123)

    W_init = np.random.uniform(size=(4, 2)).astype(np.float32)
    sess.run(tf.assign(tf_W, W_init))
    # print(sess.run(tf_W))
    bias_init = np.random.uniform(size=(2,)).astype(np.float32)
    sess.run(tf.assign(tf_bias, bias_init))

    tf_y = tf.placeholder(tf.float32, [None, 2], 'y')
    tf_loss = tf.square(tf_y - tf_out)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(tf_loss)

    for epoch in range(4):
        loss, out, _ = sess.run((tf_loss, tf_out, train_op), {tf_x: X, tf_y: y})
        if epoch % 1 == 0:
            print('epoch', epoch)
            print('loss', loss)
            print(np.argmax(out, 1))
