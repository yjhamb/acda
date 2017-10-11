'''
Denoising AutoEncoder Implementation
 
'''
import tensorflow as tf
import tensorflow.contrib.layers.fully_connected as fully_connected
import aeer.dataset.event_dataset as ds

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
    # fit the model
    fit()
    # execute and predict

if __name__ == '__main__':
    main()
    