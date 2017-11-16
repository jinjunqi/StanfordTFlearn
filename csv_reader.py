
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('..')

import tensorflow as tf

DATA_PATH = 'data/heart.csv'
BATCH_SIZE = 2
N_FEATURES = 9


def batch_generator(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(filename_queue)

    record_default = [[1.0] for _ in range(N_FEATURES)]
    record_default[4] = ['']
    record_default.append([1])

    content = tf.decode_csv(value, record_default)

    content[4] = tf.cond(tf.equal(content[4], tf.constant('Present')), lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    features = tf.stack(content[:N_FEATURES])
    label = content[-1]

    min_after_dequeue = 10 * BATCH_SIZE
    capacity = 20 * BATCH_SIZE

    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE,
                                                     capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch

def generate_batches(data_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(10):
            features, labels = sess.run([data_batch, label_batch])
            print(features)
        coord.request_stop()
        coord.join(threads)


def main():
    data_batch, label_batch = batch_generator([DATA_PATH])
    generate_batches(data_batch, label_batch)

if __name__ == '__main__':
    main()








