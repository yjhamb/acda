'''
Denoising AutoEncoder Implementation that utilizes the attention mechanism
to incorporate contextual information such as the genre for the movielens dataset
'''
import argparse
import os

import aeer.dataset.movie_dataset as ds
import tensorflow as tf

import numpy as np
from sklearn.utils import shuffle

from aeer.model.utils import ACTIVATION_FN, set_logging_config

"""
Example Usage:

Run for 20 epochs, 100 hidden units and a 0.5 corruption ratio
python attention_auto_encoder.py --epochs 20 --size 100 --corrupt 0.5

To turn off latent factors, eg for group latent factor
python attention_auto_encoder.py --nogroup
"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, default='3')
parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('-s', '--size', help='Number of hidden layer',
                    type=int, default=500)
parser.add_argument('-n', '--neg_count', help='Number of negatives', type=int,
                    default=1)
parser.add_argument('-c', '--corrupt', help='Corruption ratio', type=float,
                    default=0.2)
parser.add_argument('--save_dir', help='Directory to save the model; if not set will not save', type=str, default=None)
# Pass the Flag to disable
parser.add_argument('--nogenre', help='disable genre latent factor', action="store_true")

activation_fn_names = ACTIVATION_FN.keys()
parser.add_argument('--hidden_fn',
                    help='hidden activation function to use',
                    default='relu', type=str, choices=activation_fn_names)

parser.add_argument('--output_fn',
                    help='output activation function to use',
                    default='sigmoid', type=str, choices=activation_fn_names)


class AttentionMovieAutoEncoder(object):

    def __init__(self, n_inputs, n_hidden, n_outputs, n_genres,
                 hidden_activation='relu', output_activation='sigmoid',
                 learning_rate=0.001):
        """

        :param n_inputs: int, Number of input features (number of events)
        :param n_hidden: int, Number of hidden units
        :param n_outputs: int, Number of output features (number of events)
        :param n_genres: int, Number of genres or None to disable
        :param learning_rate: float, Step size
        """

        self.x = tf.placeholder(tf.float32, shape=[None, n_inputs])
        self.genre_id = tf.placeholder(tf.int32, shape=[None])

        # We need to gather the indices from the matrix where our outputs are
        self.gather_indices = tf.placeholder(tf.int32, shape=[None, 2])

        self.y = tf.placeholder(tf.float32, shape=[None])
        self.dropout = tf.placeholder_with_default(1.0, shape=(), name='Dropout')

        reg_constant = 0.01
        # Weights
        W = tf.get_variable('W', shape=[n_inputs, n_hidden], regularizer=tf.contrib.layers.l2_regularizer(scale=reg_constant))
        b = tf.get_variable('Bias', shape=[n_hidden])

        # Uniform Initialization U(-eps, eps)
        eps = 0.01
        preactivation = tf.nn.xw_plus_b(self.x, W, b)
        hidden = ACTIVATION_FN[hidden_activation](preactivation)
        hidden = tf.nn.dropout(hidden, self.dropout)

        attention = hidden
        # setup attention mechanism
        # Add group latent factor
        if n_genres is not None:
            genre_bias = tf.get_variable('GenreBias', shape=[n_genres, n_hidden],
                                         initializer=tf.random_uniform_initializer(-eps, eps))
            self.genre_factor = tf.nn.embedding_lookup(genre_bias, self.genre_id,
                                                       name='GenreLookup')
            g_attn_weight = tf.get_variable('AttentionWLogits',
                                                      shape=[n_hidden, 1], regularizer=tf.contrib.layers.l2_regularizer(scale=reg_constant))
            g_attention = tf.matmul(self.genre_factor, g_attn_weight)

            # Weighted sum of genre factors
            genre_weighted = tf.reduce_sum(g_attention * self.genre_factor,
                                      axis=0)
            # Add to
            attention += tf.squeeze(genre_weighted)

        attn_output = tf.nn.softmax(tf.nn.tanh(attention))

        # create the output layer
        W2 = tf.get_variable('W2', shape=[n_hidden, n_outputs], regularizer=tf.contrib.layers.l2_regularizer(scale=reg_constant))
        b2 = tf.get_variable('Bias2', shape=[n_outputs])
        preactivation_output = tf.nn.xw_plus_b(tf.multiply(attn_output, hidden), W2, b2)
        preactivation_output = tf.nn.dropout(preactivation_output, self.dropout)
        self.outputs = ACTIVATION_FN[output_activation](preactivation_output)

        self.targets = tf.gather_nd(self.outputs, self.gather_indices)
        self.actuals = tf.placeholder(tf.int64, shape=[None])

        # add weight regularizer
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # square loss
        #self.loss = tf.losses.mean_squared_error(self.targets, self.y) + sum(reg_losses)
        # square loss
        self.loss = tf.losses.mean_squared_error(self.targets, self.y)
        #self.loss = tf.losses.sigmoid_cross_entropy(self.targets, self.y)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate)
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
    if N != 0:
        precision = hits / min(N, k)
    else:
        precision = 0
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
    if N != 0:
        recall = hits / N
    else:
        recall = 0
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
    ideal_gain = 0
    topk = predictions[-N:]
    hits = 0
    # calculate the ideal gain at k
    for i in range(0, N):
        if topk[i] in actuals:
            cum_gain += 1 / np.log2(i + 2)
            hits = hits + 1

    for i in range(0, hits):
        ideal_gain += 1 / np.log2(i + 2)
    if ideal_gain != 0:
        ndcg = cum_gain / ideal_gain
    else:
        ndcg = 0
    return ndcg

def main():
    n_epochs = FLAGS.epochs
    n_hidden = FLAGS.size
    NEG_COUNT = FLAGS.neg_count
    CORRUPT_RATIO = FLAGS.corrupt

    ratings_data = ds.MovieRatingsData()
    users = ratings_data.get_train_users()

    n_inputs = ratings_data.n_movies
    n_genres = ratings_data.n_genres
    n_outputs = ratings_data.n_movies

    # We set to None to turn off the genre latent factors
    if FLAGS.nogenre:
        print("Disabling Genre Latent Factor")
        n_genres = None

    model = AttentionMovieAutoEncoder(n_inputs, n_hidden, n_outputs, n_genres,
                                    FLAGS.hidden_fn, FLAGS.output_fn,
                                    learning_rate=0.001)
    tf_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25,
                                  allow_growth=True))
    sv = tf.train.Supervisor(logdir=FLAGS.save_dir)
    set_logging_config(FLAGS.save_dir)

    with sv.prepare_or_wait_for_session(config=tf_config) as sess:
        prev_epoch_loss = 0.0
        for epoch in range(n_epochs):
            # additive gaussian noise or multiplicative mask-out/drop-out noise
            epoch_loss = 0.0
            users = shuffle(users)
            precision = []
            recall = []
            mean_avg_prec = []
            ndcg = []
            eval_at = [5, 10]

            tf.logging.info("Training the model...")
            for user_id in users:
                x, y, item = ratings_data.get_user_train_movies(
                                                    user_id, NEG_COUNT, CORRUPT_RATIO)
                train_movie_index = ratings_data.get_user_train_movie_index(user_id)
                genre_ids = ratings_data.get_user_train_genres(user_id)

                # We only compute loss on events we used as inputs
                # Each row is to index the first dimension
                gather_indices = list(zip(range(len(y)), item))

                if len(x.data) != 0:
                    # Get a batch of data
                    batch_loss, _ = sess.run([model.loss, model.train], {
                        model.x: x.toarray().astype(np.float32),
                        model.gather_indices: gather_indices,
                        model.genre_id: genre_ids,
                        model.y: y,
                        model.dropout: 0.6
                    })
                    epoch_loss += batch_loss
                    score = sess.run(model.outputs, {
                        model.x: x.toarray().astype(np.float32),
                        model.gather_indices: gather_indices,
                        model.genre_id: genre_ids,
                        model.y: y,
                        model.dropout: 1.0
                        })[0]
                    # Sorted in ascending order, we then take the last values
                    index = np.argsort(score)
    
                    # Number of test instances
                    preck = []
                    recallk = []
                    mapk = []
                    ndcgk = []
                    for k in eval_at:
                        preck.append(precision_at_k(index, train_movie_index, k))
                        recallk.append(recall_at_k(index, train_movie_index, k))
                        mapk.append(map_at_k(index, train_movie_index, k))
                        ndcgk.append(ndcg_at_k(index, train_movie_index, k))
    
                    precision.append(preck)
                    recall.append(recallk)
                    mean_avg_prec.append(mapk)
                    ndcg.append(ndcgk)
                    
            tf.logging.info("Epoch: {:>16}       Loss: {:>10,.6f}".format("%s/%s" % (epoch, n_epochs),
                                                                epoch_loss))
            tf.logging.info("")
            if prev_epoch_loss != 0 and abs(epoch_loss - prev_epoch_loss) < 1:
                tf.logging.info("Decaying learning rate...")
                model.decay_learning_rate(sess, 0.5)
            
            #prev_epoch_loss = epoch_loss
            avg_precision_5, avg_precision_10 = zip(*precision)
            avg_precision_5, avg_precision_10 = np.mean(avg_precision_5), np.mean(avg_precision_10)

            avg_recall_5, avg_recall_10 = zip(*recall)
            avg_recall_5, avg_recall_10 = np.mean(avg_recall_5), np.mean(avg_recall_10)

            avg_map_5, avg_map_10 = zip(*mean_avg_prec)
            avg_map_5, avg_map_10 = np.mean(avg_map_5), np.mean(avg_map_10)

            avg_ndcg_5, avg_ndcg_10 = zip(*ndcg)
            avg_ndcg_5, avg_ndcg_10 = np.mean(avg_ndcg_5), np.mean(avg_ndcg_10)

            # Directly access variables
            tf.logging.info(f"Precision@5: {avg_precision_5:>10.6f}       Precision@10: {avg_precision_10:>10.6f}")
            tf.logging.info(f"Recall@5:    {avg_recall_5:>10.6f}       Recall@10:    {avg_recall_10:>10.6f}")
            tf.logging.info(f"MAP@5:       {avg_map_5:>10.6f}       MAP@10:       {avg_map_10:>10.6f}")
            tf.logging.info(f"NDCG@5:      {avg_ndcg_5:>10.6f}       NDCG@10:      {avg_ndcg_10:>10.6f}")
            tf.logging.info("")

        # evaluate the model on the cv set
        cv_users = ratings_data.get_cv_users()
        precision = []
        recall = []
        mean_avg_prec = []
        ndcg = []
        eval_at = [5, 10]

        tf.logging.info("Evaluating on the CV set...")
        valid_test_users = 0
        for user_id in cv_users:
            # check if user was present in training data
            train_users = ratings_data.get_train_users()
            if user_id in train_users:
                valid_test_users = valid_test_users + 1
                test_movie_index = ratings_data.get_user_cv_movie_index(user_id)

                x, _, _ = ratings_data.get_user_train_movies(user_id, 0, 0)
                genre_ids = ratings_data.get_user_train_genres(user_id)
                # Compute score
                score = sess.run(model.outputs, {
                    model.x: x.toarray().astype(np.float32),
                    model.genre_id: genre_ids,
                    model.dropout: 1.0
                })[0] # We only do one sample at a time, take 0 index

                # Sorted in ascending order, we then take the last values
                index = np.argsort(score)

                # Number of test instances
                preck = []
                recallk = []
                mapk = []
                ndcgk = []
                for k in eval_at:
                    preck.append(precision_at_k(index, test_movie_index, k))
                    recallk.append(recall_at_k(index, test_movie_index, k))
                    mapk.append(map_at_k(index, test_movie_index, k))
                    ndcgk.append(ndcg_at_k(index, test_movie_index, k))

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
        tf.logging.info(f"Precision@5: {avg_precision_5:>10.6f}       Precision@10: {avg_precision_10:>10.6f}")
        tf.logging.info(f"Recall@5:    {avg_recall_5:>10.6f}       Recall@10:    {avg_recall_10:>10.6f}")
        tf.logging.info(f"MAP@5:       {avg_map_5:>10.6f}       MAP@10:       {avg_map_10:>10.6f}")
        tf.logging.info(f"NDCG@5:      {avg_ndcg_5:>10.6f}       NDCG@10:      {avg_ndcg_10:>10.6f}")
        tf.logging.info("")

        # evaluate on test users
        tf.logging.info("Evaluating on the test set...")
        valid_test_users = 0
        precision = []
        recall = []
        mean_avg_prec = []
        ndcg = []
        eval_at = [5, 10]
        for user_id in ratings_data.get_test_users():
            # check if user was present in training data
            train_users = ratings_data.get_train_users()
            if user_id in train_users:
                valid_test_users = valid_test_users + 1
                test_movie_index = ratings_data.get_user_test_movie_index(user_id)

                x, _, _ = ratings_data.get_user_train_movies(user_id, 0, 0)
                genre_ids = ratings_data.get_user_train_genres(user_id)
                # Compute score
                score = sess.run(model.outputs, {
                    model.x: x.toarray().astype(np.float32),
                    model.genre_id: genre_ids,
                    model.dropout: 1.0
                })[0] # We only do one sample at a time, take 0 index

                # Sorted in ascending order, we then take the last values
                index = np.argsort(score)

                # Number of test instances
                preck = []
                recallk = []
                mapk = []
                ndcgk = []
                for k in eval_at:
                    preck.append(precision_at_k(index, test_movie_index, k))
                    recallk.append(recall_at_k(index, test_movie_index, k))
                    mapk.append(map_at_k(index, test_movie_index, k))
                    ndcgk.append(ndcg_at_k(index, test_movie_index, k))

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
        tf.logging.info(f"Precision@5: {avg_precision_5:>10.6f}       Precision@10: {avg_precision_10:>10.6f}")
        tf.logging.info(f"Recall@5:    {avg_recall_5:>10.6f}       Recall@10:    {avg_recall_10:>10.6f}")
        tf.logging.info(f"MAP@5:       {avg_map_5:>10.6f}       MAP@10:       {avg_map_10:>10.6f}")
        tf.logging.info(f"NDCG@5:      {avg_ndcg_5:>10.6f}       NDCG@10:      {avg_ndcg_10:>10.6f}")
        tf.logging.info("")
    sv.request_stop()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    main()
