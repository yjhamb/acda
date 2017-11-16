'''
Denoising AutoEncoder Implementation that incorporates latent factors for
contextual group and venue data
'''
import argparse
import os

import aeer.dataset.event_dataset as ds
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

import numpy as np
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, default='3')
parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('-s', '--size', help='Number of hidden layer',
                    type=int, default=50)
parser.add_argument('-n', '--neg_count', help='Number of negatives', type=int,
                    default=4)
parser.add_argument('-c', '--corrupt', help='Corruption ratio', type=float,
                    default=0.1)
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

class LatentFactorAutoEncoder(object):

    def __init__(self, n_inputs, n_hidden, n_outputs, n_groups, n_venues,
                 learning_rate=0.001):

        self.x = tf.placeholder(tf.float32, shape=[None, n_inputs])
        self.group_id = tf.placeholder(tf.int32, shape=[None])
        self.venue_id = tf.placeholder(tf.int32, shape=[None])

        # We need to gather the indices from the matrix where our outputs are
        self.gather_indices = tf.placeholder(tf.int32, shape=[None, 2])

        self.y = tf.placeholder(tf.float32, shape=[None])

        # Weights
        W = tf.get_variable('W', shape=[n_inputs, n_hidden])
        b = tf.get_variable('Bias', shape=[n_hidden])

        # Uniform Initialization U(-eps, eps)
        eps = 0.01

        venue_bias = tf.get_variable('VenueBias', shape=[n_venues, n_hidden],
                                    initializer=tf.random_uniform_initializer(-eps, eps))
        group_bias = tf.get_variable('GroupBias', shape=[n_groups, n_hidden],
                                    initializer=tf.random_uniform_initializer(-eps, eps))

        # We lookup a bias for each
        self.venue_factor = tf.nn.embedding_lookup(venue_bias, self.venue_id,
                                                    name='VenueLookup')
        self.group_factor = tf.nn.embedding_lookup(group_bias, self.group_id,
                                                   name='GroupLookup')
        # Sum all group factors, then make it a vector so it will broadcast
        # and add it to all instances
        group_factor = tf.squeeze(tf.reduce_sum(self.group_factor, axis=0))
        venue_factor = tf.squeeze(tf.reduce_sum(self.venue_factor, axis=0))

        # Wx + b + venue + user groups
        preactivation = tf.nn.xw_plus_b(self.x, W, b) + group_factor + venue_factor

        hidden = tf.nn.relu(preactivation)

        # add weight regularizer
        # self.reg_scale = 0.01
        # self.weights_regularizer = tf.nn.l2_loss(W, "weight_loss")
        #self.reg_loss = tf.reduce_sum(tf.abs(W))

        # create the output layer with no activation function
        self.outputs = fully_connected(hidden, n_outputs, activation_fn=tf.nn.sigmoid)

        self.targets = tf.gather_nd(self.outputs, self.gather_indices)

        self.actuals = tf.placeholder(tf.int64, shape=[None])

        # evaluate metrics outputs and actuals
        self.precision_at_5 = tf.nn.in_top_k(self.outputs, self.actuals, k=5)
        self.precision_at_10 = tf.nn.in_top_k(self.outputs, self.actuals, k=10)
        self.recall_at_5 = tf.metrics.recall_at_k(self.actuals, self.outputs, k=5)
        self.recall_at_10 = tf.metrics.recall_at_k(self.actuals, self.outputs, k=10)

        # square loss
        #self.loss = tf.losses.mean_squared_error(self.targets, self.y) + self.reg_scale * self.weights_regularizer
        self.loss = tf.losses.mean_squared_error(self.targets, self.y)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Train Model
        self.train = optimizer.minimize(self.loss)


def main():
    n_epochs = FLAGS.epochs
    n_hidden = FLAGS.size
    NEG_COUNT = FLAGS.neg_count
    CORRUPT_RATIO = FLAGS.corrupt

    event_data = ds.EventData(ds.chicago_file_name)
    users = event_data.get_users()

    n_inputs = event_data.n_events
    n_groups = event_data.n_groups
    n_outputs = event_data.n_events
    n_venues = event_data.n_venues

    model = LatentFactorAutoEncoder(n_inputs, n_hidden, n_outputs, n_groups, n_venues,
                                    learning_rate=0.001)

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    tf_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                  allow_growth=True))

    with tf.Session(config=tf_config) as sess:
        init.run()
        init_local.run()
        for epoch in range(n_epochs):
            # additive gaussian noise or multiplicative mask-out/drop-out noise
            epoch_loss = 0.0
            users = shuffle(users)

            for user_id in users:
                x, y, item, group_id, venue_id = event_data.get_user_train_events_with_group(user_id, NEG_COUNT, CORRUPT_RATIO)

                # We only compute loss on events we used as inputs
                # Each row is to index the first dimension
                gather_indices = list(zip(range(len(y)), item))

                # Get a batch of data
                batch_loss, _ = sess.run([model.loss, model.train], {
                    model.x: x.toarray().astype(np.float32),
                    model.gather_indices: gather_indices,
                    model.group_id: group_id,
                    model.venue_id: venue_id,
                    model.y: y
                })

                epoch_loss += batch_loss

            print("Epoch {:,}/{:<10,} Loss: {:,.6f}".format(epoch, n_epochs,
                                                            epoch_loss))

            # evaluate the model on the test set
            test_users = event_data.get_test_users()
            precision_5 = 0
            precision_10 = 0
            recall_5 = 0
            recall_10 = 0
            valid_test_users = 0
            for user_id in test_users:
                # check if user was present in training data
                train_users = event_data.get_train_users()
                if user_id in train_users:
                    valid_test_users = valid_test_users + 1
                    #unique_user_test_events = event_data.get_user_unique_test_events(user_id)
                    test_event_index = event_data.get_user_test_event_index(user_id)
                    #[event_data._event_class_to_index[i] for i in unique_user_test_events]

                    x, _, _, group_id, venue_id = event_data.get_user_train_events_with_group(user_id, 0, 0)

                    # We replicate X, for the number of test events
                    x = np.tile(x.toarray().astype(np.float32), (len(test_event_index), 1))

                    # evaluate the model using the actuals
                    precision_at_5, precision_at_10, recall_at_5, recall_at_10 = sess.run([model.precision_at_5, 
                                                                                           model.precision_at_10,
                                                                                           model.recall_at_5,
                                                                                           model.recall_at_10], {
                        model.x: x,
                        model.actuals: test_event_index,
                        model.group_id: group_id,
                        model.venue_id: venue_id,
                    })

                    precision_5 = precision_5 + np.sum(precision_at_5)
                    precision_10 = precision_10 + np.sum(precision_at_10)
                    recall_5 = recall_5 + np.sum(recall_at_5)
                    recall_10 = recall_10 + np.sum(recall_at_10)

            avg_precision_5 = 0
            avg_precision_10 = 0
            avg_recall_5 = 0
            avg_recall_10 = 0
            if valid_test_users > 0:
                avg_precision_5 = precision_5 / valid_test_users
                avg_precision_10 = precision_10 / valid_test_users
                avg_recall_5 = recall_5 / valid_test_users
                avg_recall_10 = recall_10 / valid_test_users
                
            print("Precision@5: {:,.6f}".format(avg_precision_5))
            print("Precision@10: {:,.6f}".format(avg_precision_10))
            print("Recall@5: {:,.6f}".format(avg_recall_5))
            print("Recall@10: {:,.6f}".format(avg_recall_10))


if __name__ == '__main__':
    main()
