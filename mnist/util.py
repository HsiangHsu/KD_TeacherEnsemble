# -*- coding: utf-8 -*-
""""
Codes for KD with Teacher Ensembles
Author: Hsiang Hsu, Tica Lin
email: {hsianghsu, mlin}@g.harvard.edu
"""
import tensorflow as tf
import numpy as np
import pickle
from time import localtime, strftime
from sklearn.utils import shuffle

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def softmax_with_temperature(logits, temp=1.0, axis=1, name=None):
    logits_with_temp = logits / temp
    _softmax = tf.exp(logits_with_temp) / tf.reduce_sum(tf.exp(logits_with_temp), axis=axis, keep_dims=True)
    return _softmax
