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
from util import *


def student(x, keep_prob, num_nodes):

    # Layer 1
    W_h1 = weight_variable([784, num_nodes[0]])
    b_h1 = bias_variable([num_nodes[0]])
    h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)
    h1_drop = tf.nn.dropout(h1, keep_prob)

    # Layer 2
    W_h2 = weight_variable([num_nodes[0], num_nodes[1]])
    b_h2 = bias_variable([num_nodes[1]])
    h2 = tf.nn.relu(tf.matmul(h1_drop, W_h2) + b_h2)
    h2_drop = tf.nn.dropout(h2, keep_prob)

    # Layer 3
    # W_h3 = weight_variable([num_nodes[1], num_nodes[2]])
    # b_h3 = bias_variable([num_nodes[2]])
    # h3 = tf.nn.relu(tf.matmul(h2_drop, W_h3) + b_h3)
    # h3_drop = tf.nn.dropout(h3, keep_prob)

    # Layer 4
    # W_h4 = weight_variable([num_nodes[2], num_nodes[3]])
    # b_h4 = bias_variable([num_nodes[3]])
    # h4 = tf.nn.relu(tf.matmul(h3_drop, W_h4) + b_h4)
    # h4_drop = tf.nn.dropout(h4, keep_prob)

    # Layer 5
    W_output = tf.Variable(tf.zeros([num_nodes[1], 10]))
    b_output = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(h2_drop, W_output) + b_output

    # readout
    y = tf.nn.softmax(logits, name='output')
    y_soft_target = softmax_with_temperature(logits, temp=2.0, name='output_soft')

    return y, y_soft_target

# def compute_soft_labels(input_x, input_y, teacher_id, kp):
#     sess = tf.InteractiveSession()
#     saver = tf.train.import_meta_graph('models/model_teacher'+str(teacher_id)+'.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('models/'))
#     graph = tf.get_default_graph()
#
#     x = graph.get_tensor_by_name("x:0")
#     y_ = graph.get_tensor_by_name("y_:0")
#     keep_prob = graph.get_tensor_by_name("keep_prob:0")
#     # output_soft = graph.get_tensor_by_name("output_soft:0")
#
#     soft_labels = sess.run(output_soft, feed_dict={x: input_x, y_: input_y, keep_prob: kp})
#     sess.close()
#     return soft_labels
