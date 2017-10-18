'''
Denoising AutoEncoder Implementation

'''
from __future__ import print_function, division # Python2/3 Compatability
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import aeer.dataset.event_dataset as ds

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from scipy.stats import bernoulli
from scipy import sparse



class AutoEncoder(object):

    def __init__(self, n_inputs, n_hidden, learning_rate=0.001):
        self.x = tf.placeholder(tf.float32, shape=[None, n_inputs])

        # We need to gather the indices from the matrix where our outputs are
        self.gather_indices = tf.placeholder(tf.int32, shape=[None, 2])

        self.y = tf.placeholder(tf.float32, shape=[None])

        # Dropout inputs == Masking Noise on inputs
        # By default if we do not feed this in, no dropout will occur
        #self.dropout = tf.placeholder_with_default([1.0], None)
        n_outputs = n_inputs

        # Add some corruption
        # Add some corrpution
        # We corrupt inputs prior to feeding model
        corrupt_inputs = self.x

        # create hidden layer with default ReLU activation
        hidden = fully_connected(corrupt_inputs, n_hidden)

        # create the output layer with no activation function
        self.outputs = tf.gather_nd(
            fully_connected(hidden, n_outputs, activation_fn=None),
            self.gather_indices)

        # square loss
        self.loss = tf.losses.mean_squared_error(self.outputs,
                                                 self.y)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Train Model
        self.train = optimizer.minimize(self.loss)


def corrupt_input(x, q):
    """
    Corrupt x with probability p by setting it to 0
    else scale it by 1 / (1-p)
    """
    assert x.ndim == 1
    scale = 1.0 / (1.0-q)
    # Probability to remove it
    # p = 1; 1-p = 0
    p = 1-q
    mask = bernoulli.rvs(p, size=x.shape[0])
    # Mask outputs
    x = x * mask
    # Re-scale values
    return x * scale

def sample_negative(pos_item_map, max_items):
    """Sample uniformly items that are not observed

    :param pos_item_map: set/list, listing all of the users observed items
    :param max_items: int, item count
    :returns: int negative item id
    """
    while True:
        sample = np.random.randint(max_items)
        if sample in pos_item_map:
            continue
        return sample

def get_user_input(user_id, df, class_to_index, negative_count, corrupt_ratio):
    """
    This will get a single users input. We encode each user with a k-hot
    encoding, where a 1 if they have rated the item. We then sample
    negative items they have not observed. Negative items have a target
    of 0 and positives 1. We finally corrupt all the encoded user
    vectors.

    :param user_id: user id in dataframe
    :param df: the dataframe for training
    :param class_to_index: dictionary that maps item ids to indices
    :param negative_count: int, number of negative samples
    :param corrupt_ratio: float, [0, 1] the probability of corrupting samples
    :returns: Encoded User Vector, Y Target, item ids
    """

    item_count = len(class_to_index)

    # Get all positive items
    positives = [class_to_index[i] for i in df.eventId[df.memberId == user_id].unique()]

    # Sample negative items
    negatives = [sample_negative(positives, item_count) for _ in range(negative_count)]

    input_count = len(positives) + len(negatives)

    # X vector for a single user
    # Duplicate input count times
    x_data = [1.0] * input_count

    # Indices for the items
    cols = positives + negatives
    rows = []
    for i in range(input_count):
        rows.extend([i] * input_count)

    x = sparse.coo_matrix((x_data * input_count,
                           (rows, cols * input_count)),
                          shape=(input_count, item_count),
                          dtype=np.float32)

    # Negative targets are 0, positives are 1
    y_targets = np.zeros(input_count, dtype=np.float32)
    y_targets[:len(positives)] = 1.0

    # Sparse Matrix; directly take the data and corrupt it
    x.data = corrupt_input(x.data, corrupt_ratio).astype(np.float32)
    return x, y_targets, cols

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
    train_x, test_x, _, _ = event_data.split_dataset()

    def get_batch(df, user_ids, mlb):
        """
        This method creates a single vector for each user where all of their
        events are set to a 1, otherwise its a 0.

        In this case, this user has observed events: 1 and 4
        [1, 0, 0, 1, 0]

        TODO: Should probably abstract this out somewhere

        :param df: pd.DataFrame of test/train
        :param user_ids: list of user ids, this will query the dataframe
        :param mlb: sklearn.MultiLabelBinarizer fitted with the event data
        :returns: np.array of values
        """
        item_ids = [df.eventId[df.memberId == uid].unique() for uid in user_ids
                    if len(df.eventId[df.memberId == uid].unique()) > 0]
        return mlb.transform(item_ids), item_ids

    init = tf.global_variables_initializer()

    n_epochs = 10
    batch_size = 64
    batches_per_iteration = int(len(users) / batch_size)

    NEG_COUNT = 4
    CORRUPT_RATIO = 0.5

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            # additive gaussian noise or multiplicative mask-out/drop-out noise
            epoch_loss = 0.0
            users = shuffle(users)

            for user_id in users:
                x, y, item = get_user_input(user_id, train_x, class_to_index, NEG_COUNT, CORRUPT_RATIO)

                # We only compute loss on events we used as inputs
                # Each row is to index the first dimension
                gather_indices = zip(range(len(y)), item)

                # Get a batch of data
                batch_loss, _ = sess.run([model.loss, model.train], {
                    model.x: x.toarray().astype(np.float32),
                    model.gather_indices: gather_indices,
                    model.y: y
                })

            # for idx in range(batches_per_iteration):
            #     # Batch indices
            #     lo = batch_size * idx
            #     hi = lo + batch_size

            #     x, item_ids = get_batch(train_x, users[lo:hi], mlb)

            #     # We only compute loss on events we used as inputs
            #     gather_indices = []
            #     for row_index, ids in enumerate(item_ids):
            #         gather_indices.extend([[row_index, class_to_index[iid]]
            #                                for iid in ids])

            #     # Get a batch of data
            #     batch_loss, _ = sess.run([model.loss, model.train], {
            #         model.x: x,
            #         model.gather_indices: gather_indices,
            #         #model.dropout: 0.5 # Dropout = Some masking noise
            #     })
                epoch_loss += batch_loss

            print("Epoch {:,}/{:<10,} Loss: {:,.6f}".format(epoch, n_epochs,
                                                            epoch_loss))
         
            # evaluate the model on the test set
            for user_id in users:
                # check if user was present in training data
                if user_id in train_x.memberId:
                    x, y, item = get_user_input(user_id, test_x, class_to_index, NEG_COUNT, CORRUPT_RATIO)

                    # We only compute loss on events we used as inputs
                    # Each row is to index the first dimension
                    gather_indices = zip(range(len(y)), item)

                    # Get a batch of data
                    batch_loss, _ = sess.run([model.loss, model.train], {
                        model.x: x.toarray().astype(np.float32),
                        model.gather_indices: gather_indices,
                        model.y: y
                        })
                    
            
if __name__ == '__main__':
    main()