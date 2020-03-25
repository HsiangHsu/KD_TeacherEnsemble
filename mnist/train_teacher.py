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
from teacher_models import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Random seeds
seed = 9
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

filename = 'train_teacher'+str(seed)
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

# trainning parameters
n_epochs = 50
batch_size = 50
num_nodes = [1200, 1200]
learning_rate = 1e-3

n_batches = len(mnist.train.images) // batch_size

# placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

y, y_soft_target = teacher(x, keep_prob, num_nodes)

# Loss functions
loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

losses = []
accs = []
test_accs = []

saver = tf.train.Saver()

file.write('Training\n')
file.flush()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        x_shuffle, y_shuffle = shuffle(mnist.train.images, mnist.train.labels)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x, batch_y = x_shuffle[start:end], y_shuffle[start:end]
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob:0.5})
        train_loss = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, keep_prob:0.5})
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})

        if epoch % 10 == 0:
            file.write("Epoch : {}, Loss : {:.4f}, Accuracy: {:.4f}, Test accuracy: {:.4f}\n".format(epoch+1, train_loss, train_accuracy, test_accuracy))
            file.flush()

        losses.append(train_loss)
        accs.append(train_accuracy)
        test_accs.append(test_accuracy)

    saver.save(sess, "models/model_teacher"+str(seed))

file.write("... completed!\n")
file.flush()
file.close()



# file.write('Saving Results\n')
# file.flush()
# f = open('mnist_diff.pickle', 'wb')
# save = {
#     'idx': nodes,
#     'imgs': X[nodes]
#     }
# pickle.dump(save, f, 2)
# f.close()
#
# file.close()
