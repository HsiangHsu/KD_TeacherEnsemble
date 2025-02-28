import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from utils import cross_entropy_loss

class DeepModel():
    def __init__(self,inputs, num_classes, temperature):
        config = {
            'weight_decay': 1e-3,
            'conv1_size': 10,
            'mp1_size': 3,
            'conv2_size': 20,
            'mp2_size': 3,
            'conv3_size': 30,
            'conv4_size': 50,
            'mp4_size': 3,
            'fc5_size': 50,
            'out_size': 10
        }

        optimization_config = {
            'learning_rate': 0.1,
            'decay_steps': 50,
            'decay_rate': 0.96,
        }
        _, H, W, C = inputs.shape
        self.X = tf.placeholder(tf.float32, [None, H, W, C], name='X_placeholder')
        self.teacher_logits = tf.placeholder(tf.float32, [None, num_classes], name='Y_placeholder')

        net = self._create_5x5_conv_layers(config)
        net = self._create_3x3_conv_layer(net, config)
        self.logits = self._create_fully_connected_layers(net, config)

        # self.loss = tf.losses.mean_squared_error(self.teacher_logits, self.logits)
        self.loss = cross_entropy_loss(self.logits / temperature, self.teacher_logits / temperature)
        self.optimization = self._create_optimization(optimization_config)

    def _create_5x5_conv_layers(self, config):
        net = None
        with tf.contrib.framework.arg_scope([layers.convolution2d],
            kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=layers.l2_regularizer(config['weight_decay'])):

            net = layers.convolution2d(self.X, config['conv1_size'], scope='st-conv1', data_format='NHWC')
            net = tf.layers.batch_normalization(net)
            net = layers.max_pool2d(net, config['mp1_size'], scope='st-mp1', data_format='NHWC')
            net = layers.convolution2d(net, config['conv2_size'], scope='st-conv2', data_format='NHWC')
            net = tf.layers.batch_normalization(net)
            net = layers.max_pool2d(net, config['mp2_size'], scope='st-mp2', data_format='NHWC')
        return net

    def _create_3x3_conv_layer(self, net, config):
        with tf.contrib.framework.arg_scope([layers.convolution2d],
            kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=layers.l2_regularizer(config['weight_decay'])):

            net = layers.convolution2d(net, config['conv3_size'], scope='st-conv3', data_format='NHWC')
            net = tf.layers.batch_normalization(net)
            net = layers.convolution2d(net, config['conv4_size'], scope='st-conv4', data_format='NHWC')
            net = tf.layers.batch_normalization(net)
            net = layers.max_pool2d(net, config['mp4_size'], scope='st-mp4', data_format='NHWC')
        return net

    def _create_fully_connected_layers(self, net, config):
        with tf.contrib.framework.arg_scope([layers.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=layers.l2_regularizer(config['weight_decay'])):

            net = layers.flatten(net)
            net = layers.fully_connected(net, config['fc5_size'], scope='st-fc5')
            net = tf.layers.batch_normalization(net)
            net = layers.fully_connected(net, config['out_size'], activation_fn=None, scope='st-logits')
        return net

    def _create_optimization(self, config):
        self.global_step = tf.placeholder(tf.int32, [])
        self.learning_rate = tf.train.exponential_decay(
            config['learning_rate'], self.global_step, config['decay_steps'], config['decay_rate'])
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

    def train(self, train_x, session, config, teacher, callback_fn):
        batch_size = config['batch_size']
        max_epochs = config['max_epochs']
        self._check_batch_size(train_x, batch_size)
        num_examples = train_x.shape[0]
        num_batches = num_examples // batch_size
        for epoch in range(1, max_epochs + 1):
            train_x = self._permutate_dataset(train_x)
            total_loss = 0
            for i in range(num_batches):
                loss_value = self._learn_on_batch(session, teacher, train_x, batch_size, i, epoch)
                total_loss += loss_value
            print(f"epoch {epoch}, average loss: {total_loss / num_examples}")
            callback_fn(session, config, self)

    def _check_batch_size(self, x, batch_size):
        num_examples = x.shape[0]
        assert num_examples % batch_size == 0

    def _permutate_dataset(self, x):
        num_examples = x.shape[0]
        permutation_idx = np.random.permutation(num_examples)
        return x[permutation_idx]

    def _learn_on_batch(self, session, teacher, train_x, batch_size, index, epoch):
        batch_x = self._create_batch(train_x, batch_size, index)
        teacher_logits = np.squeeze(self._get_teacher_logits(session, teacher, batch_x))
        loss_value, _ = session.run(
            [self.loss, self.optimization],
            feed_dict={self.X: batch_x, self.teacher_logits: teacher_logits, self.global_step: epoch})
        return loss_value

    def _create_batch(self, set, batch_size, index):
        return set[index * batch_size : (index + 1) * batch_size, ...]

    def _get_teacher_logits(self, session, teacher, x):
        teacher_soft_labels = np.zeros((len(teacher), 50, 10))
        for i in range(len(teacher)):
            lo = session.run(teacher[i].logits, feed_dict={teacher[i].X: x})
            teacher_soft_labels[i, :, :]=lo
        logits = np.mean(teacher_soft_labels, axis=0)
        # [logits] = session.run([teacher.logits], feed_dict={teacher.X: x})
       
        return logits