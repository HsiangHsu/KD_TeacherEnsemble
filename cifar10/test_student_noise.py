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

filename = 'cifar10/test_student'
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

# num_teacher_list = [1, 2, 5, 10]
num_teacher_list = [1, 2]
noise_sigma_list = [0, .1, .2, .5, 1., 2., 5.]

train_acc = np.zeros((len(num_teacher_list), len(noise_sigma_list)))
test_acc = np.zeros((len(num_teacher_list), len(noise_sigma_list)))


for i in range(len(num_teacher_list)):
    num_teacher = num_teacher_list[i]
    file.write('Teacher {}\n'.format(num_teacher))
    file.flush()
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('cifar10/models/model_student'+str(num_teacher)+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models/'))
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    # output_soft = graph.get_tensor_by_name("output_soft:0")

    for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        file.write(' Noise Variance: {:.4f}\n'.format(sigma))
        train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images+np.random.normal(0, sigma, mnist.train.images.shape), y_: mnist.train.labels, keep_prob:1.0})
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images+np.random.normal(0, sigma, mnist.test.images.shape), y_: mnist.test.labels, keep_prob:1.0})
        file.write('  Train ACC: {:.4f}, Test ACC: {:.4f}\n'.format(train_accuracy, test_accuracy))
        file.flush()

        train_acc[i, j] = train_accuracy
        test_acc[i, j] = test_accuracy

        # soft_labels = sess.run(output_soft, feed_dict={x: mnist.train.images+np.random.normal(0, sigma, mnist.train.images.shape), keep_prob:1.0})
        # f = open('teacher_outputs_'+str(i)+'_'+str(sigma)+'.pickle', 'wb')
        # save = {
        #     'soft_labels': soft_labels
        #     }
        # pickle.dump(save, f, 2)
        # f.close()

    sess.close()
file.close()


f = open('cifar10/mnist_student_noise.pickle', 'wb')
save = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc
    }
pickle.dump(save, f, 2)
f.close()
