import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.model_selection import train_test_split

import os
from datetime import datetime
import cv2
import cnn_model
import data_manipulation

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Parameters
training_epochs = 50
TRAIN_BATCH_SIZE = 128
display_step = 100
TEST_BATCH_SIZE = 128


def read_data():
    images = []
    for filename in os.listdir('C:/Users/Furkan/Desktop/Bitirme/dataset/images'):
        img = cv2.imread(os.path.join('C:/Users/Furkan/Desktop/Bitirme/dataset/images', filename))
        if img is not None:
            # For labelling manually.
            # cv2.imshow("Frame", img)
            # cv2.waitKey(0)
            images.append(img)

    return images


def read_labels():
    data = pd.read_csv('C:/Users/Furkan/Desktop/Bitirme/dataset/images/Labels.csv')
    X = data.iloc[0:, 1].values
    label_list = X

    return label_list


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


def train(f_maps, conv_stride, maxp_stride, training_epochs):
    batch_size = TRAIN_BATCH_SIZE
    num_labels = 5

    data = read_data()
    labels = read_labels()

    print('Applying contrast streching manipulation...')
    contrasted_data, contrasted_labels = data_manipulation.apply_contrast(data, labels)
    contrasted_data = np.asarray(contrasted_data)
    print('Size of contrasted data: %d' % len(contrasted_data))
    print('Shape of contrasted data: %s' % str(contrasted_data.shape))

    print('Applying rotation manipulation...')
    rotated_data, rotated_labels = data_manipulation.apply_rotation(contrasted_data, labels)
    rotated_data = np.asarray(rotated_data)
    print('Size of rotated data: %d' % len(rotated_data))
    print('Shape of rotated data: %s' % str(rotated_data.shape))

    print('Applying shifting manipulation...')
    shifted_data, shifted_labels = data_manipulation.apply_shifting(contrasted_data, labels)
    shifted_data = np.asarray(shifted_data)
    print('Size of shifted data: %d' % len(shifted_data))
    print('Shape of shifted data: %s' % str(shifted_data.shape))

    print('Applying flipping manipulation...')
    flipped_data, flipped_labels = data_manipulation.apply_horizontal_flip(contrasted_data, labels)
    flipped_data = np.asarray(flipped_data)
    print('Size of flipped data: %d' % len(flipped_data))
    print('Shape of flipped data: %s' % str(flipped_data.shape))

    print('Concatenating manipulated data')
    concat_data = np.concatenate((rotated_data, contrasted_data, flipped_data, shifted_data), axis=0)
    print("Size of total data: %s" % str(len(concat_data)))

    print("Size of labels: %s" % str(len(labels)))
    from sklearn.preprocessing import LabelBinarizer
    le = LabelBinarizer()
    encoded_labels = le.fit_transform(labels)

    concat_data = np.asarray(concat_data)
    print("Shape of data: %s" % str(concat_data.shape))
    concat_data = concat_data.reshape(-1, 4800)

    print("Size of encoded labels: %s" % str(encoded_labels))

    X_train, X_test, y_train, y_test = split_dataset(concat_data, encoded_labels[:len(concat_data)])
    print("Size of data: %s" % str(X_train.shape))
    print("Length of data: %s" % str(len(X_train)))
    print("Size of labels: %s" % str(y_train.shape))
    print("Length of labels: %s" % str(len(y_train)))
    total_train_data = np.concatenate((X_train, y_train), axis=1)
    print("Size of total data: %s" % str(total_train_data.shape))
    train_size = total_train_data.shape[0]
    validation_data = X_test
    validation_labels = y_test

    is_training = tf.placeholder(tf.bool, name='MODE')

    # Tensorflow variables should be initialized.
    x = tf.placeholder(tf.float32, [None, 4800])
    y_ = tf.placeholder(tf.float32, [None, 5])

    y = cnn_model.CNN(x, feature_maps=f_maps, conv_stride=conv_stride, maxp_stride=maxp_stride)

    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y, y_)

    tf.summary.scalar('loss', loss)

    with tf.name_scope("ADAM"):
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(1e-4, batch * batch_size, train_size, 0.95, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('acc', accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    total_batch = int(train_size / batch_size)

    # Writing a log file is optional
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        np.random.shuffle(total_train_data)
        train_data_ = total_train_data[:, :-num_labels]
        train_labels_ = total_train_data[:, -num_labels:]

        for i in range(total_batch):
            offset = (i * batch_size) % train_size
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op],
                                                  feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Optional
            summary_writer.add_summary(summary, epoch * total_batch + i)

            if i % display_step == 0:
                format_str = '%s: step %d, accuracy = %.3f'
                print(format_str % (datetime.now(), (epoch+1), train_accuracy))

    saver.save(sess, MODEL_DIRECTORY)

    test_size = validation_labels.shape[0]
    batch_size = TEST_BATCH_SIZE
    total_batch = int(test_size / batch_size)

    saver.restore(sess, MODEL_DIRECTORY)

    acc_buffer = []

    for i in range(total_batch):
        offset = (i * batch_size) % test_size
        batch_xs = validation_data[offset:(offset + batch_size), :]
        batch_ys = validation_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
        correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))
        acc_buffer.append(np.sum(correct_prediction) / batch_size)

    print("Test accuracy for this model: %g" % np.mean(acc_buffer))


def main():
    train(64, [5, 5], [2, 2], training_epochs)

if __name__ == '__main__':
    main()
