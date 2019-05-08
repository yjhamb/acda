'''
Denoising AutoEncoder Implementation that utilizes the attention mechanism
to incorporate contextual information such as the group and venue
'''
"""
Example Usage:

Run for 20 epochs, 100 hidden units and a 0.5 corruption ratio
python attention_auto_encoder.py --epochs 20 --size 100 --corrupt 0.5

To turn off latent factors, eg for group latent factor
python attention_auto_encoder.py --nogroup
"""

from common.utils import ACTIVATION_FN
import tensorflow as tf

class AttentionAutoEncoder(object):

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
        self.dropout = tf.placeholder_with_default(1.0, shape=(), name='Dropout')

        # Weights
        W = tf.get_variable('W', shape=[n_inputs, n_hidden])
        b = tf.get_variable('Bias', shape=[n_hidden])

        # Uniform Initialization U(-eps, eps)
        eps = 0.01

        preactivation = tf.nn.xw_plus_b(self.x, W, b)
        hidden = ACTIVATION_FN[hidden_activation](preactivation)
        hidden = tf.nn.dropout(hidden, self.dropout)

        attention = hidden
        # setup attention mechanism
        # Add venue latent factor
        if n_venues is not None:
            # Create and lookup each bias
            venue_bias = tf.get_variable('VenueBias', shape=[n_venues, n_hidden],
                                         initializer=tf.random_uniform_initializer(-eps, eps))
            self.venue_factor = tf.nn.embedding_lookup(venue_bias, self.venue_id,
                                                       name='VenueLookup')
            v_attn_weight = tf.get_variable('V_AttentionWLogits',
                                                      shape=[n_hidden, 1])
            v_attention = tf.matmul(self.venue_factor, v_attn_weight)

            # Weighted sum of venue factors
            venue_weighted = tf.reduce_sum(v_attention * self.venue_factor,
                                      axis=0)
            # Sum all venue factors, then make it a vector so it will broadcast
            # and add it to all instances
            attention += tf.squeeze(venue_weighted)

        # Add group latent factor
        if n_groups is not None:
            group_bias = tf.get_variable('GroupBias', shape=[n_groups, n_hidden],
                                         initializer=tf.random_uniform_initializer(-eps, eps))
            self.group_factor = tf.nn.embedding_lookup(group_bias, self.group_id,
                                                       name='GroupLookup')
            g_attn_weight = tf.get_variable('AttentionWLogits',
                                                      shape=[n_hidden, 1])
            g_attention = tf.matmul(self.group_factor, g_attn_weight)

            # Weighted sum of group factors
            group_weighted = tf.reduce_sum(g_attention * self.group_factor,
                                      axis=0)
            # Add to
            attention += tf.squeeze(group_weighted)

        attn_output = tf.nn.softmax(tf.nn.tanh(attention))

        # create the output layer
        W2 = tf.get_variable('W2', shape=[n_hidden, n_outputs])
        b2 = tf.get_variable('Bias2', shape=[n_outputs])
        preactivation_output = tf.nn.xw_plus_b(tf.multiply(attn_output, hidden), W2, b2)
        
        self.outputs = ACTIVATION_FN[output_activation](preactivation_output)

        self.targets = tf.gather_nd(self.outputs, self.gather_indices)
        self.actuals = tf.placeholder(tf.int64, shape=[None])

        self.loss = tf.losses.mean_squared_error(self.targets, self.y)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Train Model
        self.train = optimizer.minimize(self.loss)
