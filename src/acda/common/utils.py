import logging
from logging.config import dictConfig
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

def set_logging_config(save_directory=None):
    """
    Setup the logging configuration to a file if necessary to store the logs
    from tensorflow.

    :param save_directory: string, directory to store the log files
    """
    logger = logging.getLogger('tensorflow')
    handlers = logger.handlers
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s")
    handlers[0].setFormatter(formatter)

    # Setup Logging
    config = dict(
        version=1,
        formatters={
            # For files
            'detailed': {
                'format': "[%(asctime)s - %(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
            },
            # For the console
            'console': {
                'format':
                "[%(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
            }
        },
        disable_existing_loggers=False
    )

    # Update if we use a file
    if save_directory:
        file_handler = logging.FileHandler("{}/log".format(save_directory))
        detailed_format = logging.Formatter("[%(asctime)s - %(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s")
        file_handler.setFormatter(detailed_format)
        # Add file hanlder to tensorflow logger
        logger.addHandler(file_handler)

    dictConfig(config)
