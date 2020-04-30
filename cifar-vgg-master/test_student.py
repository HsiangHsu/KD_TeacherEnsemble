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
# from sklearn.utils import shuffle
# from util import *
# from teacher_models import *

import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from numpy import savetxt

filename = 'test_student'
file = open(filename+'_log.txt','w')
file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
file.flush()

noise_sigma_list = [0, 1, 2, 5, 10, 20, 50]
# noise_sigma_list = [0, .1, .2, .5, 1., 2., 5.]
number_of_teacher = 10
num_classes = 10
train_acc = np.zeros((number_of_teacher, len(noise_sigma_list)))
test_acc = np.zeros((number_of_teacher, len(noise_sigma_list)))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


for i in range(number_of_teacher):
    file.write('Student {}\n'.format(i))
    file.flush()
    # sess = tf.InteractiveSession()
    # saver = tf.train.import_meta_graph('models/model_teacher'+str(i)+'.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('models/'))
    # graph = tf.get_default_graph()

    # x = graph.get_tensor_by_name("x:0")
    # y_ = graph.get_tensor_by_name("y_:0")
    # keep_prob = graph.get_tensor_by_name("keep_prob:0")
    # accuracy = graph.get_tensor_by_name("accuracy:0")
    # output_soft = graph.get_tensor_by_name("output_soft:0")

    model = load_model('models/model_student'+str(i+1)+'.h5')
    print("load model ", i+1)
    model.summary()
    #  #training parameters
    # batch_size = 128
    # maxepoches = 10
    # learning_rate = 0.1
    # lr_decay = 1e-6
    # lr_drop = 20


    model.fit(x=x_test, y=y_test, validation_data=(x_test, y_test))
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1]) 


    for j in range(len(noise_sigma_list)):
        sigma = noise_sigma_list[j]
        file.write(' Noise Variance: {:.4f}\n'.format(sigma))

        train_scores = model.evaluate(x_train + np.random.normal(0, sigma, x_train.shape), y_train, verbose=2)
        test_scores = model.evaluate(x_test + np.random.normal(0, sigma, x_test.shape), y_test, verbose=2)
        train_accuracy = train_scores[1]
        test_accuracy = test_scores[1]
        
        # train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images+np.random.normal(0, sigma, mnist.train.images.shape), y_: mnist.train.labels, keep_prob:1.0})
        # test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images+np.random.normal(0, sigma, mnist.test.images.shape), y_: mnist.test.labels, keep_prob:1.0})
        file.write('  Train ACC: {:.4f}, Test ACC: {:.4f}\n'.format(train_accuracy, test_accuracy))
        file.flush()

        train_acc[i, j] = train_accuracy
        test_acc[i, j] = test_accuracy
        
        print('  Train ACC: {:.4f}, Test ACC: {:.4f}\n'.format(train_accuracy, test_accuracy))
        # soft_labels = sess.run(output_soft, feed_dict={x: mnist.train.images+np.random.normal(0, sigma, mnist.train.images.shape), keep_prob:1.0})
        # f = open('teacher_outputs_'+str(i)+'_'+str(sigma)+'.pickle', 'wb')
        # save = {
        #     'soft_labels': soft_labels
        #     }
        # pickle.dump(save, f, 2)
        # f.close()

    # sess.close()
savetxt('student_train_acc.csv', train_acc, delimiter=',')
savetxt('student_test_acc.csv', test_acc, delimiter=',')
file.close()


f = open('cifar_student_noise.pickle', 'wb')
save = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc
    }
pickle.dump(save, f, 2)
f.close()
