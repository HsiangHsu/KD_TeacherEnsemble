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
from load_data import *

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# load the saved dataset
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
test_features, test_labels = pickle.load(open('preprocess_testing.p', mode='rb'))

filename = 'cifar10/test_teacher'
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

noise_sigma_list = [0, .1, .2, .5, 1., 2., 5.]
number_of_teacher = 9 #change
train_acc = np.zeros((number_of_teacher, len(noise_sigma_list)))
test_acc = np.zeros((number_of_teacher, len(noise_sigma_list)))

for i in range(number_of_teacher):
    file.write('Teacher {}\n'.format(i))
    file.flush()
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('cifar10/models/model_teacher'+str(i)+'.meta')
    saver.restore(sess, 'cifar10/models/model_teacher'+str(i))
    graph = tf.get_default_graph()
   
    x = graph.get_tensor_by_name("input_x:0")
    y_ = graph.get_tensor_by_name("output_y:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    output_soft = graph.get_tensor_by_name("logits:0")


    for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        file.write(' Noise Variance: {:.4f}\n'.format(sigma))
        # train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images+np.random.normal(0, sigma, mnist.train.images.shape), y_: mnist.train.labels, keep_prob:1.0})
        # test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images+np.random.normal(0, sigma, mnist.test.images.shape), y_: mnist.test.labels, keep_prob:1.0})
        
        # train_accuracy=0
        # soft_labels=np.array([])
        # n_batches=5 
        # for batch_i in range(1, n_batches + 1):
        #     batch_features, batch_labels = pickle.load(open('preprocess_batch_' + str(batch_i) + '.p', mode='rb'))
        #     batch_acc=sess.run(accuracy, feed_dict={x: batch_features+np.random.normal(0, sigma, batch_features.shape), y_: batch_labels, keep_prob:1.0})
        #     train_accuracy=train_accuracy + batch_acc
        #     batch_labels = sess.run(output_soft, feed_dict={x: batch_features+np.random.normal(0, sigma, batch_features.shape), keep_prob:1.0})
        #     soft_labels= np.concatenate((soft_labels, batch_labels))
        
        #     print(soft_labels.shape, batch_labels.shape)

        # train_accuracy=train_accuracy/n_batches
        
        train_accuracy = sess.run(accuracy, feed_dict={x: valid_features+np.random.normal(0, sigma, valid_features.shape), y_: valid_labels, keep_prob:1.0})
        test_accuracy = sess.run(accuracy, feed_dict={x: test_features+np.random.normal(0, sigma, test_features.shape), y_: test_labels, keep_prob:1.0})
        print(' Noise Variance: {:.4f}  Train ACC: {:.4f}, Test ACC: {:.4f}\n'.format(sigma,train_accuracy, test_accuracy))
        file.write('  Train ACC: {:.4f}, Test ACC: {:.4f}\n'.format(train_accuracy, test_accuracy))
        file.flush()

        train_acc[i, j] = train_accuracy
        test_acc[i, j] = test_accuracy

        soft_labels = sess.run(output_soft, feed_dict={x: valid_features+np.random.normal(0, sigma, valid_features.shape), keep_prob:1.0})
        f = open('cifar10/output/teacher_outputs_'+str(i)+'_'+str(sigma)+'.pickle', 'wb')
        save = {
            'soft_labels': soft_labels
            }
        pickle.dump(save, f, 2)
        f.close()

    sess.close()
file.close()


f = open('cifar10/cifar_teacher_noise.pickle', 'wb')
save = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc
    }
pickle.dump(save, f, 2)
f.close()
