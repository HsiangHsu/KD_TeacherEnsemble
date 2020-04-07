# -*- coding: utf-8 -*-
""""
Codes for KD with Teacher Ensembles
Author: Hsiang Hsu, Tica Lin
email: {hsianghsu, mlin}@g.harvard.edu
"""
import tensorflow as tf
import numpy as np
import pickle
import random
import sys
from time import localtime, strftime
from sklearn.utils import shuffle
from util import *
from teacher_models import *
from load_data import *

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
from os.path import isfile, isdir
import tarfile
from urllib.request import urlretrieve
from os.path import isfile, isdir

# Random seeds
# seed = int(sys.argv[1])
seed=8
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

filename = 'cifar10/train_teacher'+str(seed)
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()


# Explore the dataset
batch_id = 3
sample_id = 7000
display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

# Preprocess all the data and save it
preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

# load the saved dataset
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# Hyper parameters
epochs = 10
batch_size = 128
keep_probability = 0.7
learning_rate = 0.001

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Build model
logits = conv_net(x, keep_prob)
model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# Training Phase
file.write('Training\n')
file.flush()
print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, valid_features, valid_labels, accuracy)

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, "cifar10/models/model_teacher"+str(seed))
    

file.write("... completed!\n")
file.flush()
file.close()
