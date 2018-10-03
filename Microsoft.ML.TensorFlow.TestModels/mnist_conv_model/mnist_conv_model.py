import os
import sys
import time
from datetime import datetime
import math
import pandas as pd
import numpy as np
import tensorflow as tf

def load_data(fileName):
    data = pd.read_csv(fileName, header = None, sep='\t')
    colnames = [ 'V'+str(x) for x in range(0, data.shape[1])]
    data.columns = colnames
    return data

def train_model(dataTrain, labCol, config):
    print('Training Data Dimensions: (%d,%d)' % (dataTrain.shape[0],dataTrain.shape[1]))
    colNames = np.array(list(dataTrain))
    features = np.delete(colNames,labCol)
    train_X = dataTrain.ix[:, features].values
    train_X = train_X.reshape(train_X.shape[0], 28,28, 1)
    train_Y = dataTrain.ix[:,labCol].values.ravel()
    
    
    tf.set_random_seed(1)
    lr = tf.placeholder(tf.float32, name = "learning_rate")
    pkeep = tf.placeholder_with_default(1.0, shape=(), name="DropoutProb")
    features = tf.placeholder(tf.float32, [None, train_X.shape[1], train_X.shape[2], train_X.shape[3]], name="Features")
    labels = tf.placeholder(tf.int64, [None], "Label")
    
    K = 6  # first convolutional layer output depth
    L = 12  # second convolutional layer output depth
    M = 24  # third convolutional layer
    N = 200  # fully connected layer (softmax)

    W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))

    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(features, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, pkeep)
    model = tf.matmul(YY4, W5) + B5
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model), name='Loss')

    prediction = tf.nn.softmax(model, name = "Prediction")
    accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(prediction,1), labels), tf.float32), name = "Accuracy")
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, name="MomentumOp")

    init = tf.global_variables_initializer()
    # Launch the graph.
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    #sess = tf.Session()
    sess.run(init)

    batch_size =config['batch_size']
    train_time_sec = 0
    total_batch = int(train_X.shape[0] / batch_size)
    for epoch in range(config['epochs']):
        avg_loss = 0
        perm = np.arange(train_X.shape[0])
        np.random.shuffle(perm)
        train_X = train_X[perm] 
        train_Y = train_Y[perm] 
        for batch_idx in range(0, train_X.shape[0], batch_size):
            X_batch = train_X[batch_idx:batch_idx+batch_size]
            Y_batch = train_Y[batch_idx:batch_idx+batch_size]
            t0 = time.time()
            _, loss_val, acc = sess.run([optimizer, loss, accuracy], feed_dict={features: X_batch, labels: Y_batch, pkeep:0.9, lr:0.01})
            train_time_sec = train_time_sec + (time.time() - t0)
            avg_loss += loss_val / total_batch
        print('Epoch: ', '%04d' % (epoch+1), 'cost (cross-entropy) = %.4f , acc = %.4f' % (avg_loss, acc))

    tf.saved_model.simple_save(
       sess,
       export_dir = "./conv",
       inputs = {"Features" : features},
       outputs = {"Prediction": prediction})


tool_version = tf.__version__
print('TensorFlow version: {}.'.format(tool_version))

config = {}
config['batch_size'] = 100
config['epochs'] = 20
config['model_dir'] = os.getcwd()+'/model'

# Start the clock!
ptm  = time.time()

print('Loading data...')
dataTrain = load_data('mnist_train.1K.tsv')
dataTest = load_data('mnist_test.1K.tsv')
print('Done!\n')

labCol = [0]
modelData = train_model(dataTrain, labCol, config)
