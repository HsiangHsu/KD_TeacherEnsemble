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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Random seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

num_teacher = 10

filename = 'train_student'+str(num_teacher)
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

# trainning parameters
n_epochs = 50
batch_size = 50
num_nodes = [600, 600]
learning_rate = 1e-3

n_batches = len(mnist.train.images) // batch_size

# placeholders
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
soft_target_ = tf.placeholder(tf.float32, [None, 10], name='soft_target_')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
T = tf.placeholder(tf.float32, name='T')

y, y_soft_target = student(x, keep_prob, num_nodes)

# Loss functions
loss_hard_target = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
loss_soft_target = -tf.reduce_sum(soft_target_ * tf.log(y_soft_target), reduction_indices=[1])
loss = tf.reduce_mean(tf.square(T) * loss_hard_target + tf.square(T) * loss_soft_target)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

losses = []
accs = []
test_accs = []

saver = tf.train.Saver()

file.write('Training\n')
file.flush()


sigma = 0

# soft_targets  = compute_soft_labels(mnist.train.images, mnist.train.labels, teacher_id, 1.0)
# tf.reset_default_graph()

teacher_soft_labels = np.zeros((num_teacher, 55000, 10))
for i in range(num_teacher):
    pickle_file = 'teacher_outputs_'+str(i)+'_'+str(sigma)+'.pickle'
    with open(pickle_file, "rb") as input_file:
        teacher_soft_labels[i, :, :] = pickle.load(input_file, encoding="latin1")['soft_labels']

soft_targets = np.mean(teacher_soft_labels, axis=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        x_shuffle, y_shuffle, soft_targets_shuffle = shuffle(mnist.train.images, mnist.train.labels, soft_targets)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x, batch_y, batch_soft_targets = x_shuffle[start:end], y_shuffle[start:end], soft_targets_shuffle[start:end]
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, soft_target_:batch_soft_targets,keep_prob:0.5, T:2.0})

        if epoch % 10 == 0:
            train_loss = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, soft_target_:batch_soft_targets, keep_prob:0.5, T:2.0})
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0, T:1.0})
            test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0, T:1.0})
            file.write("Epoch : {}, Loss : {:.4f}, Accuracy: {:.4f}, Test accuracy: {:.4f}\n".format(epoch+1, train_loss, train_accuracy, test_accuracy))
            file.flush()

    saver.save(sess, "models/model_student"+str(num_teacher))

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
