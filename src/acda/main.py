'''
Main driver program
'''

import argparse
import os

from sklearn.utils import shuffle

from common.metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k
from common.utils import ACTIVATION_FN, set_logging_config
import dataset.event_dataset as ds
import dataset.user_group_dataset as ug_dataset
from model.event_adae import AttentionAutoEncoder
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, default='3')
parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('-s', '--size', help='Number of hidden layer',
                    type=int, default=100)
parser.add_argument('-n', '--neg_count', help='Number of negatives', type=int,
                    default=4)
parser.add_argument('-c', '--corrupt', help='Corruption ratio', type=float,
                    default=0.1)
parser.add_argument('--save_dir', help='Directory to save the model; if not set will not save', type=str, default=None)
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

    model = AttentionAutoEncoder(n_inputs, n_hidden, n_outputs, n_groups, n_venues,
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

            tf.logging.info("Training the model...")
            for user_id in users:
                x, y, item = event_data.get_user_train_events(
                                                    user_id, NEG_COUNT, CORRUPT_RATIO)
                train_event_index = event_data.get_user_train_event_index(user_id)
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
                    model.y: y,
                    model.dropout: 0.8
                })
                epoch_loss += batch_loss
                
            tf.logging.info("Epoch: {:>16}       Loss: {:>10,.6f}".format("%s/%s" % (epoch, n_epochs),
                                                                epoch_loss))
            tf.logging.info("")
            if prev_epoch_loss != 0 and abs(epoch_loss - prev_epoch_loss) < 1:
                tf.logging.info("Decaying learning rate...")
                model.decay_learning_rate(sess, 0.5)
        
        # evaluate the model on the cv set
        cv_users = event_data.get_cv_users()
        precision = []
        recall = []
        mean_avg_prec = []
        ndcg = []
        eval_at = [5, 10]

        tf.logging.info("Evaluating on the CV set...")
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
                    model.dropout: 1.0
                })[0]  # We only do one sample at a time, take 0 index

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
        for user_id in event_data.get_test_users():
            # check if user was present in training data
            train_users = event_data.get_train_users()
            if user_id in train_users:
                valid_test_users = valid_test_users + 1
                test_event_index = event_data.get_user_test_event_index(user_id)

                x, _, _ = event_data.get_user_train_events(user_id, 0, 0)
                group_ids = event_data.get_user_train_groups(user_id)
                venue_ids = event_data.get_user_train_venues(user_id)
                # Compute score
                score = sess.run(model.outputs, {
                    model.x: x.toarray().astype(np.float32),
                    model.group_id: group_ids,
                    model.venue_id: venue_ids,
                    model.dropout: 1.0
                })[0]  # We only do one sample at a time, take 0 index

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