import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01
batch_size = 128
n_epochs = 30

mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y_placeholder')

w = tf.Variable(tf.random_normal(shape=[784, 10], seed=0.01), name='weight')
b = tf.Variable(tf.zeros(shape=[1, 10]), name='bias')

logits = X * w + b
entroy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entroy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)

    for i in range(n_epochs):
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch

        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('Total time: {0}'.format(time.time() - start_time))
    print('Optimization Finished ! ')

    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    n_batches = int(mnist.test.num_example/batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_example))

    writer.close()




