from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 20
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784], name="Features") # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10], name = "Label") # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.constant(0.1, shape=[784, 10]), name = "W")
b = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32), name = "b")

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b, name = "Prediction") # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1), name="Loss")
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name = "SGDOptimizer")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver_def = tf.train.Saver().as_saver_def()
# Start training
with tf.Session() as sess:

    # Run the initializer
   sess.run(init)
   tf.saved_model.simple_save(
    sess,
    export_dir = "./lr",
    inputs = {"Features" : x},
    outputs = {"Prediction": pred})