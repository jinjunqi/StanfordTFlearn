import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows -1

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

Y_predicted = X * w + b
loss = tf.square(Y - Y_predicted, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)

tf.summary.scalar("loss", loss)
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    for i in range(100):
        total_loss = 0
        for x, y in data:
            # _, lo = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
            # total_loss += lo
            _, summary = sess.run([optimizer, merged_summary], feed_dict={X: x, Y: y})
            writer.add_summary(summary=summary, global_step=i)
        # print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()

    w, b = sess.run([w, b])


X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X*w+b, 'r', label='Predicted data')
plt.legend()
plt.show()



