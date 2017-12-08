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

from aeer.model.utils import ACTIVATION_FN
import aeer.dataset.user_group_dataset as ug_dataset

"""
Example Usage:

Run for 20 epochs, 100 hidden units and a 0.5 corruption ratio
python latent_auto_encoder.py --epochs 20 --size 100 --corrupt 0.5

To turn off latent factors, eg for group latent factor
python latent_auto_encoder.py --nogroup
"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, default='3')
parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('-s', '--size', help='Number of hidden layer',
                    type=int, default=100)
parser.add_argument('-n', '--neg_count', help='Number of negatives', type=int,
                    default=4)
parser.add_argument('-c', '--corrupt', help='Corruption ratio', type=float,
                    default=0.1)
# Pass the Flag to disable
parser.add_argument('--nogroup', help='disable group latent factor', action="store_true")
parser.add_argument('--novenue', help='disable venue latent factor', action="store_true")

activation_fn_names = ACTIVATION_FN.keys()
parser.add_argument('--hidden_fn',
                    help='hidden activation function to use',
                    default='relu', type=str, choices=activation_fn_names)

parser.add_argument('--output_fn',
                    help='output activation function to use',
                    default='sigmoid', type=str, choices=activation_fn_names)


class LatentFactorAutoEncoder(object):

    def __init__(self, n_inputs, n_hidden, n_outputs, n_groups, n_venues,
                 hidden_activation='relu', output_activation='sigmoid',
                 learning_rate=0.001):
        """

        :param n_inputs: int, Number of input features (number of events)
        :param n_hidden: int, Number of hidden units
        :param n_outputs: int, Number of output features (number of events)
        :param n_groups: int, Number of groups or None to disable
        :param n_venues: int, Number of venues or None to disable
        :param learning_rate: float, Step size
        """
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

        # Wx + b + venue + user groups
        preactivation = tf.nn.xw_plus_b(self.x, W, b)

        # Add venue latent factor
        if n_venues is not None:
            # Create and lookup each bias
            venue_bias = tf.get_variable('VenueBias', shape=[n_venues, n_hidden],
                                         initializer=tf.random_uniform_initializer(-eps, eps))
            self.venue_factor = tf.nn.embedding_lookup(venue_bias, self.venue_id,
                                                       name='VenueLookup')
            v_attn_weight = tf.get_variable('V_AttentionWLogits',
                                                      shape=[n_hidden, 1])
            # Multiply Group Factors by a weight vector
            # We could make this a deeper network...
            v_attention = tf.nn.softmax(tf.matmul(self.venue_factor, v_attn_weight))

            # Weighted sum of venue factors
            venue_weighted = tf.reduce_sum(v_attention * self.venue_factor,
                                      axis=0)
            # Sum all venue factors, then make it a vector so it will broadcast
            # and add it to all instances
            preactivation += tf.squeeze(venue_weighted)

        # Add group latent factor
        if n_groups is not None:
            group_bias = tf.get_variable('GroupBias', shape=[n_groups, n_hidden],
                                         initializer=tf.random_uniform_initializer(-eps, eps))
            self.group_factor = tf.nn.embedding_lookup(group_bias, self.group_id,
                                                       name='GroupLookup')
            attn_weight = tf.get_variable('AttentionWLogits',
                                                      shape=[n_hidden, 1])
            # Multiply Group Factors by a weight vector
            # We could make this a deeper network...
            attention = tf.nn.softmax(tf.matmul(self.group_factor, attn_weight))

            # Weighted sum of group factors
            group_weighted = tf.reduce_sum(attention * self.group_factor,
                                      axis=0)
            # Add to
            preactivation += tf.squeeze(group_weighted)

        hidden = ACTIVATION_FN[hidden_activation](preactivation)
        # add weight regularizer
        # self.reg_scale = 0.01
        # self.weights_regularizer = tf.nn.l2_loss(W, "weight_loss")
        #self.reg_loss = tf.reduce_sum(tf.abs(W))

        # create the second hidden layer
        #hidden2 = fully_connected(hidden, n_hidden,
        #                               activation_fn=ACTIVATION_FN[hidden_activation])
        #W2 = tf.get_variable('W2', shape=[n_hidden, n_hidden])
        #b2 = tf.get_variable('Bias2', shape=[n_hidden])
        #preactivation2 = tf.nn.xw_plus_b(hidden, W2, b2)
        
        #preactivation2 += tf.squeeze(group_weighted)
        
        #hidden2 = ACTIVATION_FN[hidden_activation](preactivation2)

        # create the output layer
        self.outputs = fully_connected(hidden, n_outputs,
                                       activation_fn=ACTIVATION_FN[output_activation])

        self.targets = tf.gather_nd(self.outputs, self.gather_indices)

        self.actuals = tf.placeholder(tf.int64, shape=[None])

        # square loss
        #self.loss = tf.losses.mean_squared_error(self.targets, self.y) + self.reg_scale * self.weights_regularizer
        self.loss = tf.losses.mean_squared_error(self.targets, self.y)
        #self.loss = tf.losses.log_loss(self.targets, self.y)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Train Model
        self.train = optimizer.minimize(self.loss)

def precision_at_k(predictions, actuals, k):
    """
    Computes the precision at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns precision: float, the precision score at k
    """
    N = len(actuals)
    hits = len(set(predictions[-k:]).intersection(set(actuals)))
    precision = hits / min(N, k)
    return precision

def recall_at_k(predictions, actuals, k):
    """
    Computes the recall at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns recall: float, the recall score at k
    """
    N = len(actuals)
    hits = len(set(predictions[-k:]).intersection(set(actuals)))
    recall = hits / N
    return recall

def map_at_k(predictions, actuals, k):
    """
    Computes the MAP at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns MAP: float, the score at k
    """
    avg_prec = []
    for i in range(1, k+1):
        prec = precision_at_k(predictions, actuals, i)
        avg_prec.append(prec)
    return np.mean(avg_prec)

def ndcg_at_k(predictions, actuals, k):
    """
    Computes the NDCG at k
    :param predictions: array, predicted values
    :param actuals: array, actual values
    :param k: int, value to compute the metric at
    :returns NDCG: float, the score at k
    """
    N = min(len(actuals), k)
    cum_gain = 0
    ideal_gain = 1
    topk = predictions[-N:]
    if topk[0] in actuals:
        cum_gain = 1
    # calculate the ideal gain at k
    for i in range(2, N):
        ideal_gain += 1 / np.log2(i)
        if topk[i] in actuals:
            cum_gain += 1 / np.log2(i)

    return cum_gain / ideal_gain

def main():
    n_epochs = FLAGS.epochs
    n_hidden = FLAGS.size
    NEG_COUNT = FLAGS.neg_count
    CORRUPT_RATIO = FLAGS.corrupt

    event_data = ds.EventData(ds.rsvp_chicago_file, ug_dataset.user_group_chicago_file)
    users = event_data.get_train_users()

    n_inputs = event_data.n_events
    n_groups = event_data.n_groups
    n_outputs = event_data.n_events
    n_venues = event_data.n_venues

    # We set to None to turn off the group/venue latent factors
    if FLAGS.nogroup:
        print("Disabling Group Latent Factor")
        n_groups = None

    if FLAGS.novenue:
        print("Disabling Venue Latent Factor")
        n_venues = None

    model = LatentFactorAutoEncoder(n_inputs, n_hidden, n_outputs, n_groups, n_venues,
                                    FLAGS.hidden_fn, FLAGS.output_fn,
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
                x, y, item = event_data.get_user_train_events(
                                                    user_id, NEG_COUNT, CORRUPT_RATIO)

                group_ids = event_data.get_user_train_groups(user_id)
                venue_ids = event_data.get_user_train_venues(user_id)

                # We only compute loss on events we used as inputs
                # Each row is to index the first dimension
                gather_indices = list(zip(range(len(y)), item))

                # Get a batch of data
                batch_loss, _ = sess.run([model.loss, model.train], {
                    model.x: x.toarray().astype(np.float32),
                    model.gather_indices: gather_indices,
                    model.group_id: group_ids,
                    model.venue_id: venue_ids,
                    model.y: y
                })
                epoch_loss += batch_loss
            print("Epoch: {:>16}       Loss: {:>10,.6f}".format("%s/%s" % (epoch, n_epochs),
                                                                epoch_loss))

            # evaluate the model on the test set
            cv_users = event_data.get_cv_users()
            precision = []
            recall = []
            mean_avg_prec = []
            ndcg = []
            eval_at = [5, 10]

            valid_test_users = 0
            for user_id in cv_users:
                # check if user was present in training data
                train_users = event_data.get_train_users()
                if user_id in train_users:
                    valid_test_users = valid_test_users + 1
                    test_event_index = event_data.get_user_cv_event_index(user_id)

                    x, _, _ = event_data.get_user_train_events(user_id, 0, 0)
                    group_ids = event_data.get_user_train_groups(user_id)
                    venue_ids = event_data.get_user_train_venues(user_id)
                    # Compute score
                    score = sess.run(model.outputs, {
                        model.x: x.toarray().astype(np.float32),
                        model.group_id: group_ids,
                        model.venue_id: venue_ids,
                    })[0] # We only do one sample at a time, take 0 index

                    # Sorted in ascending order, we then take the last values
                    index = np.argsort(score)

                    # Number of test instances
                    preck = []
                    recallk = []
                    mapk = []
                    ndcgk = []
                    for k in eval_at:
                        preck.append(precision_at_k(index, test_event_index, k))
                        recallk.append(recall_at_k(index, test_event_index, k))
                        mapk.append(map_at_k(index, test_event_index, k))
                        ndcgk.append(ndcg_at_k(index, test_event_index, k))

                    precision.append(preck)
                    recall.append(recallk)
                    mean_avg_prec.append(mapk)
                    ndcg.append(ndcgk)

            if valid_test_users > 0:
                # Unpack the lists zip(*[[1,2], [3, 4]]) => [1,3], [2,4]
                avg_precision_5, avg_precision_10 = zip(*precision)
                avg_precision_5, avg_precision_10 = np.mean(avg_precision_5), np.mean(avg_precision_10)

                avg_recall_5, avg_recall_10 = zip(*recall)
                avg_recall_5, avg_recall_10 = np.mean(avg_recall_5), np.mean(avg_recall_10)

                avg_map_5, avg_map_10 = zip(*mean_avg_prec)
                avg_map_5, avg_map_10 = np.mean(avg_map_5), np.mean(avg_map_10)

                avg_ndcg_5, avg_ndcg_10 = zip(*ndcg)
                avg_ndcg_5, avg_ndcg_10 = np.mean(avg_ndcg_5), np.mean(avg_ndcg_10)

            # Directly access variables
            print(f"Precision@5: {avg_precision_5:>10.6f}       Precision@10: {avg_precision_10:>10.6f}")
            print(f"Recall@5:    {avg_recall_5:>10.6f}       Recall@10:    {avg_recall_10:>10.6f}")
            print(f"MAP@5:       {avg_map_5:>10.6f}       MAP@10:       {avg_map_10:>10.6f}")
            print(f"NDCG@5:      {avg_ndcg_5:>10.6f}       NDCG@10:      {avg_ndcg_10:>10.6f}")
            print()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    main()
