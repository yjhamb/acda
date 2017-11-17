import tensorflow as tf

# List of activation function mappings
ACTIVATION_FN = {
    'elu': tf.nn.elu, # Exponential Linear Unit
    'relu6': tf.nn.relu6,
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'sigmoid': tf.nn.sigmoid,
    'identity': tf.identity,
    'softplus': tf.nn.softplus,
    'softsign': tf.nn.softsign,
}
