'''
Denoising AutoEncoder Implementation

'''
from __future__ import print_function, division  # Python2/3 Compatability
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import aeer.dataset.event_dataset as ds
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle


class AutoEncoder(object):

    def __init__(self, n_inputs, n_hidden, learning_rate=0.001):
        self.x = tf.placeholder(tf.float32, shape=[None, n_inputs])

        # We need to gather the indices from the matrix where our outputs are
        self.gather_indices = tf.placeholder(tf.int32, shape=[None, 2])

        self.y = tf.placeholder(tf.float32, shape=[None])

        # Dropout inputs == Masking Noise on inputs
        # By default if we do not feed this in, no dropout will occur
        # self.dropout = tf.placeholder_with_default([1.0], None)
        n_outputs = n_inputs

        # create hidden layer with default ReLU activation
        hidden = fully_connected(self.x, n_hidden)

        # create the output layer with no activation function
        self.outputs = fully_connected(hidden, n_outputs, activation_fn=None)
        
        self.targets = tf.gather_nd(self.outputs, self.gather_indices)
        
        self.actuals = tf.placeholder(tf.int32, shape=[n_outputs])

        # evaluate top k wrt outputs and actuals
        self.top_k = tf.nn.in_top_k(self.outputs, self.actuals, k=10)

        # square loss
        self.loss = tf.losses.mean_squared_error(self.targets,
                                                 self.y)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Train Model
        self.train = optimizer.minimize(self.loss)


def main():
    event_data = ds.EventData(ds.chicago_file_name)
    users = event_data.get_users()
    events = event_data.get_events()

    # Convert the sparse event indices to a dense vector
    mlb = MultiLabelBinarizer()
    mlb.fit([events])
    # We need this to get the indices of events
    class_to_index = dict(zip(mlb.classes_, range(len(mlb.classes_))))

    model = AutoEncoder(len(events), 50, learning_rate=0.001)

    init = tf.global_variables_initializer()

    n_epochs = 1
    NEG_COUNT = 4
    CORRUPT_RATIO = 0.5

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            # additive gaussian noise or multiplicative mask-out/drop-out noise
            epoch_loss = 0.0
            users = shuffle(users)

            for user_id in users:
                x, y, item = event_data.get_user_train_events(user_id, class_to_index, NEG_COUNT, CORRUPT_RATIO)

                # We only compute loss on events we used as inputs
                # Each row is to index the first dimension
                gather_indices = list(zip(range(len(y)), item))

                # Get a batch of data
                batch_loss, _ = sess.run([model.loss, model.train], {
                    model.x: x.toarray().astype(np.float32),
                    model.gather_indices: gather_indices,
                    model.y: y
                })

                epoch_loss += batch_loss

            print("Epoch {:,}/{:<10,} Loss: {:,.6f}".format(epoch, n_epochs,
                                                            epoch_loss))

        # evaluate the model on the test set
        test_users = event_data.get_test_users()
        precision = 0
        valid_test_users = 0
        for user_id in test_users:
            # check if user was present in training data
            train_users = event_data.get_train_users()
            if user_id in train_users:
                valid_test_users = valid_test_users + 1
                unique_user_test_events = event_data.get_user_unique_test_events(user_id)
                x, _, _ = event_data.get_user_test_events(user_id, class_to_index)

                # evaluate the model using the actuals
                top_k_events = sess.run(model.top_k, {
                    model.x: x.toarray().astype(np.float32),
                    model.actuals: unique_user_test_events
                })

                precision = precision + np.sum(top_k_events)

        avg_precision = 0
        if (valid_test_users > 0):
            avg_precision = precision / valid_test_users

        print("Precision: {:,.6f}".format(avg_precision))

if __name__ == '__main__':
    main()
