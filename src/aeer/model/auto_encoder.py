'''
Denoising AutoEncoder Implementation

'''
from __future__ import print_function, division # Python2/3 Compatability
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import aeer.dataset.event_dataset as ds

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle


class AutoEncoder(object):

    def __init__(self, n_inputs, n_hidden, learning_rate=0.001):
        self.x = tf.placeholder(tf.float32, shape=[None, n_inputs])

        # Dropout inputs == Masking Noise on inputs
        # By default if we do not feed this in, no dropout will occur
        self.dropout = tf.placeholder_with_default([1.0], None)

        n_outputs = n_inputs

        # Add some corrpution
        corrupt_inputs = tf.nn.dropout(self.x, self.dropout)

        # create hidden layer with default ReLU activation
        hidden = fully_connected(corrupt_inputs, n_hidden)

        # create the output layer with no activation function
        outputs = fully_connected(hidden, n_outputs, activation_fn=None)

        # square loss
        self.loss = tf.losses.mean_squared_error(outputs, self.x)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Train Model
        self.train = optimizer.minimize(self.loss)


def fit():
    # get the number of event count
    n_inputs = ds.get_event_count()
    n_hidden = 50
    n_outputs = n_inputs
    learning_rate = 0.01

    x = tf.placeholder(tf.float32, shape=[None, n_inputs])
    # create hidden layer with default ReLU activation
    hidden = fully_connected(x, n_hidden);
    # create the output layer with no activation function
    outputs = fully_connected(hidden, n_outputs, activation_fn=None);
    # square loss
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - x))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)

    # evaluate
    # tf.nn.in_top_k
    # can only use log loss or cross-entropy loss
    # value range has to be -1 to 1 for

    init = tf.global_variables_initializer()
    # save the model
    saver = tf.train.Saver()

    n_epochs = 400
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            # feed the training data
            # additive gaussian noise or multiplicative mask-out/drop-out noise
            sess.run(training_op)

        # save_path = saver.save(sess, "/path to file")



def main():
    event_data = ds.EventData(ds.chicago_file_name)
    users = event_data.get_users()
    events = event_data.get_events()

    # Convert the sparse event indices to a dense vector
    mlb = MultiLabelBinarizer()
    mlb.fit([events])

    model = AutoEncoder(len(events), 50, learning_rate=0.001)
    train_x, test_x, train_y, test_y = event_data.split_dataset()

    def get_batch(df, user_ids, mlb):
        """
        This method creates a single vector for each user where all of their
        events are set to a 1, otherwise its a 0.
        eg

        In this case, this user has observed events: 1 and 4
        [1, 0, 0, 1, 0]

        TODO: Should probably abstract this out somewhere

        :param df: pd.DataFrame of test/train
        :param user_ids: list of user ids, this will query the dataframe
        :param mlb: sklearn.MultiLabelBinarizer fitted with the event data
        :returns: np.array of values
        """
        item_ids = [df.eventId[df.memberId == uid].unique() for uid in user_ids]
        return mlb.transform(item_ids)

    init = tf.global_variables_initializer()

    n_epochs = 400
    batch_size = 64
    batches_per_iteration = int(len(users) / batch_size)

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            # additive gaussian noise or multiplicative mask-out/drop-out noise
            epoch_loss = 0.0
            users = shuffle(users)
            for idx in range(batches_per_iteration):
                # Batch indices
                lo = batch_size * idx
                hi = lo + batch_size
                # Get a batch of data
                batch_loss, _ = sess.run([model.loss, model.train], {
                    model.x: get_batch(train_x, users[lo:hi], mlb),
                    model.dropout: 0.5 # Dropout = Some masking noise
                })
                epoch_loss += batch_loss

            print("Epoch {:,}/{:<10,} Loss: {:,.6f}".format(epoch, n_epochs,
                                                            epoch_loss))

if __name__ == '__main__':
    main()
