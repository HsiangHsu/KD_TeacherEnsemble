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

filename = 'test_teacher'
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
    file.write('Teacher {}\n'.format(i))
    file.flush()
    
    model = load_model('models/model_teacher'+str(i+1)+'.h5')
    print("load model ", i+1)
    model.summary()
    
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
        
        file.write('  Train ACC: {:.4f}, Test ACC: {:.4f}\n'.format(train_accuracy, test_accuracy))
        file.flush()

        train_acc[i, j] = train_accuracy
        test_acc[i, j] = test_accuracy

        print('  Train ACC: {:.4f}, Test ACC: {:.4f}\n'.format(train_accuracy, test_accuracy))
        
        
file.close()


f = open('cifar_teacher_noise.pickle', 'wb')
save = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc
    }
pickle.dump(save, f, 2)
f.close()
