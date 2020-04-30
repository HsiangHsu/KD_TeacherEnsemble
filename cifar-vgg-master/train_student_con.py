# -*- coding: utf-8 -*-
""""
Codes for KD with Teacher Ensembles
Author: Hsiang Hsu, Tica Lin
email: {hsianghsu, mlin}@g.harvard.edu
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
import numpy as np
from numpy import array, expand_dims, not_equal
from tensorflow.keras.datasets import cifar10
from numpy import dstack
from student_model import *
from time import localtime, strftime
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16

batch_size = 128
maxepoches = 180
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
n_members = 10



# load models from file
def load_all_models(n_models, x_test, y_test):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'models/model_teacher' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        model.fit(x=x_test, y=y_test, validation_data=(x_test, y_test),verbose=2)
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models
 
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    # make predictions
    yhats = [model.predict(inputX) for model in members]
    yhats = array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = argmax(summed, axis=1)
    return result
 
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, x_train, y_train, x_test, y_test):
    # create dataset using ensemble
    stackedY = stacked_dataset(members, x_train)
    y_stack = to_categorical(stackedY, 10)
    # fit standalone model
    # model = LogisticRegression()
    input_shape = (32,32,3)
    model_input = Input(shape=input_shape)

    VGGmodel = VGG16(weights='imagenet', include_top=False)
    VGG_output = VGGmodel(model_input)

    m_main = Flatten(name='flatten_main')(VGG_output)
    m_main = Dense(4096, activation='relu', name='fc1_main')(m_main)
    m_main = Dense(4096, activation='relu', name='fc2_main')(m_main)
    m_main = Dense(10, activation='softmax', name='main_output')(m_main)

    m_ensemble = Flatten(name='flatten_ensemble')(VGG_output)
    m_ensemble = Dense(4096, activation='relu', name='fc1_ensemble')(m_ensemble)
    m_ensemble = Dense(4096, activation='relu', name='fc2_ensemble')(m_ensemble)
    m_ensemble = Dense(10, activation='softmax', name='ensemble_output')(m_ensemble)

    model = Model(inputs=model_input, outputs=[m_main,  m_ensemble])
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

    model.compile(loss={'main_output': 'categorical_crossentropy', 
                    'ensemble_output': 'categorical_crossentropy'},
              loss_weights={'main_output': 1.0,
                            'ensemble_output': 0.2},
              optimizer=sgd,
              metrics=['accuracy'])

    model.fit(x_train, 
          {'main_output': y_train, 'ensemble_output': y_stack},
          batch_size=batch_size,
          epochs=maxepoches,
          validation_data= (x_test, {'main_output': y_test, 'ensemble_output': y_test}),
          verbose=2)

    model.save_weights('model_student'+str(n_members)+'.h5')
    model.save("models/model_student"+str(n_members)+".h5")

    return model
 

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(x_train.shape, x_test.shape)

    # load all models
    
    members = load_all_models(n_members, x_test, y_test)
    print('Loaded %d models' % len(members))

    filename = 'train_student'
    file = open(filename+'_log_'+str(n_members)+'.txt','w')
    file.write(strftime("%Y-%m-%d-%H.%M.%S\n", localtime()))
    file.write('Number of Teacher {}\n'.format(n_members))
    file.flush()
    # for model in members:
    #     _, acc = model.evaluate(x_test, testy_enc, verbose=0)
    #     print('Model Accuracy: %.3f' % acc)
    
    # fit stacked model using the ensemble
    model = fit_stacked_model(members, x_train, y_train, x_test, y_test)
    

    file.close()
    
    

    
    