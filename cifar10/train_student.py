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
from time import localtime, strftime
from sklearn.utils import shuffle
from util import *
from student_models import *
from load_data import *

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# load the saved dataset
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
test_features, test_labels = pickle.load(open('preprocess_testing.p', mode='rb'))

# Random seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

num_teacher = 9

filename = 'cifar10/train_student'+str(num_teacher)
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

# Hyper parameters
epochs = 50
# batch_size = 128
keep_probability = 0.7
learning_rate = 0.001

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# placeholders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
soft_target_ = tf.placeholder(tf.float32, shape=[None, 10], name='soft_target_')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
T = tf.placeholder(tf.float32, name='T')


# Build model (student)
logits = conv_net(x, keep_prob)
model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


saver = tf.train.Saver()

# Training Phase
file.write('Training\n')
file.flush()
print('Training...')

sigma = 0

# soft_targets  = compute_soft_labels(mnist.train.images, mnist.train.labels, teacher_id, 1.0)
# tf.reset_default_graph()

teacher_soft_labels = np.zeros((num_teacher, 5000, 10))
for i in range(num_teacher):
    pickle_file = 'cifar10/output/teacher_outputs_'+str(i)+'_'+str(sigma)+'.pickle'
    with open(pickle_file, "rb") as input_file:
        teacher_soft_labels[i, :, :] = pickle.load(input_file, encoding="latin1")['soft_labels']

soft_targets = np.mean(teacher_soft_labels, axis=0)


with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Use only validation set instead of looping over all batches
        sess.run(optimizer, feed_dict={x: valid_features, y: valid_labels, soft_target_:soft_targets,keep_prob:keep_probability, T:2.0})

       
        train_loss = sess.run(cost, feed_dict={x: valid_features, y: valid_labels, soft_target_:soft_targets, keep_prob:keep_probability, T:2.0})
        train_accuracy = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob:1.0, T:1.0})
        test_accuracy = sess.run(accuracy, feed_dict={x: test_features, y: test_labels, keep_prob:1.0, T:1.0})
        file.write("Epoch : {}, Loss : {:.4f}, Accuracy: {:.4f}, Test accuracy: {:.4f}\n".format(epoch+1, train_loss, train_accuracy, test_accuracy))
        file.flush()

        print("Epoch : {}, Loss : {:.4f}, Accuracy: {:.4f}, Test accuracy: {:.4f}\n".format(epoch+1, train_loss, train_accuracy, test_accuracy))
        # print_stats(sess, valid_features, valid_labels, cost, test_features, test_labels, accuracy)

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, "cifar10/models/model_student"+str(num_teacher))
    

file.write("... completed!\n")
file.flush()
file.close()


