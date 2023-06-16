#!/usr/bin/env python3

""" Mini-Batch """


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        batch_size=32,
        epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt"):
    ''' trains a loaded neural network model using mini-batch 
    gradient descent'''
    # Define the neural network model

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + '.meta')
        new_saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for i in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            if i < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for j in range(0, X_train.shape[0], batch_size):
                    X_batch = X_shuffled[j:j + batch_size]
                    Y_batch = Y_shuffled[j:j + batch_size]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                    if j != 0 and (j / batch_size + 1) % 100 == 0:
                        batch_cost = sess.run(
                            loss, feed_dict={x: X_batch, y: Y_batch})
                        batch_accuracy = sess.run(
                            accuracy, feed_dict={x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(j // batch_size + 1))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy: {}".format(batch_accuracy))
        return new_saver.save(sess, save_path)
