#!/usr/bin/env python3

""" Mini-Batch """


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    ''' trains a loaded neural network model using mini-batch gradient descent'''
    # Define the neural network model

    with tf.Session() as sess:
        ''' import meta graph and restore weights '''
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        graph = tf.get_default_graph()

        ''' get placeholders from graph '''
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        m = graph.get_tensor_by_name("m:0")
        accuracy = graph.get_tensor_by_name("Mean_1:0")
        loss = graph.get_tensor_by_name("Mean:0")
        train_op = graph.get_operation_by_name("train_op")

        ''' get collection of trainable variables '''
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        for epoch in range(epochs + 1):
            steps = m // batch_size + 1
            print("After {} epochs:".format(epoch))
            train_cost = sess.run(
                loss, feed_dict={x: X_train, y: Y_train, m: X_train.shape[0]})
            print("\tTraining Cost: {}".format(train_cost))
            train_accuracy = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train, m: X_train.shape[0]})
            print("\tTraining Accuracy: {}".format(train_accuracy))
            valid_cost = sess.run(
                loss, feed_dict={x: X_valid, y: Y_valid, m: X_valid.shape[0]})
            print("\tValidation Cost: {}".format(valid_cost))
            valid_accuracy = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid, m: X_valid.shape[0]})
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            x_shuffled = X_train
            y_shuffled = Y_train

            ''' shuffle training data '''
            for step in range(steps):
                star = step * batch_size
                end = (step + 1) * batch_size
                x_batch = x_shuffled[star:end]
                y_batch = y_shuffled[star:end]

                sess.run(train_op, feed_dict={
                         x: x_batch, y: y_batch, m: x_batch.shape[0]})
                if step != 0 and (step + 1) % 100 == 0:
                    step_cost = sess.run(
                        loss, feed_dict={x: x_batch, y: y_batch, m: x_batch.shape[0]})
                    step_accuracy = sess.run(accuracy, feed_dict={
                                             x: x_batch, y: y_batch, m: x_batch.shape[0]})
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
        ''' save the model '''
        return saver.save(sess, save_path)
