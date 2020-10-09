import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu
from utils.network_summary import count_parameters
from tensorflow.contrib.framework import arg_scope
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import classifier_utils as utils

from skimage.util import random_noise
import sys

# sys.path.append("./models/research/slim")
# from nets.mobilenet import mobilenet_v2
slim = tf.contrib.slim


# from popular_models.nets.resnet_v1 import resnet_v1_50
# from popular_models.nets import resnet_v1


###### pretrained resnet18  #######

class ResNet18(object):
    def __init__(self, is_train, reuse_weights=False):
        self._reuse_weights = reuse_weights
        self._counted_scope = []
        self._flops = 0
        self._weights = 0
        self.is_train = is_train
        self._global_step = tf.Variable(0, trainable=False, name='global_step')

    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h / stride) * (w / stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def __call__(self, images, labels, classes):
        print('Building model')
        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(images, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            # x = self._fc(x, 128, name='fc1')
            x = self._fc(x, classes, name='fc2')

        logits = x

        # Probs & preds & acc
        probs = tf.nn.softmax(x)
        preds = tf.to_int32(tf.argmax(logits, 1))
        ones = tf.constant(np.ones([int(images.get_shape()[0])]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([int(images.get_shape()[0])]), dtype=tf.float32)
        labels_index = tf.to_int32(tf.argmax(labels, 1))
        correct = tf.where(tf.equal(preds, labels_index), ones, zeros)
        acc = tf.reduce_mean(correct)

        # Loss & acc
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=labels_index)
        loss = tf.reduce_mean(losses)

        ###### training all weights
        # variables = tf.global_variables()
        # self.variables = variables

        ###### training fc weights
        self.variables = tf.contrib.framework.get_variables("logits/")

        return loss, acc


# ##### pretrained_resnet
class pretrained_resnet_architecture:
    def __init__(self, name='resnet_classifier'):
        self.reuse = tf.AUTO_REUSE
        self.name = name

    def __call__(self, inputs, input_y, classes, training):

        # with tf.variable_scope('resnet_v1_50', reuse = self.reuse):
        if inputs.get_shape()[3] < 3:
            inputs_three_channel = tf.concat([inputs, inputs, inputs], axis=3)
        else:
            inputs_three_channel = inputs

        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.00001)):
            logits, _ = resnet_v1_50(inputs_three_channel,
                                     num_classes=classes,
                                     is_training=True)
            logits = tf.squeeze(logits)

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # feature = tf.squeeze(feature, axis=[1, 2])
            # logits = slim.fully_connected(feature, num_outputs=classes, activation_fn=None, scope='Predict')
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits))
            # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.variables = tf.contrib.framework.get_variables("resnet_v1_50/logits") + tf.contrib.framework.get_variables(
            "resnet_v1_50/AuxLogits")
        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        count_parameters(self.variables, name="classifier_parameter_num")
        return cost, accuracy


# ##### pretrained_resnet


####### pretrained_resnet50
# class pretrained_resnet_architecture:
#   def __init__(self, name='resnet_classifier'):
#     self.reuse = tf.AUTO_REUSE
#     self.name = name
#   def __call__(self,inputs,input_y, classes, training):
#     # with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.00001),reuse=self.reuse):
#     with tf.variable_scope(self.name, reuse = self.reuse):
#         inputs_three_channel = tf.concat([inputs, inputs, inputs], axis=3)
#         logits, _ = resnet_v1_50(inputs_three_channel,num_classes=classes,is_training=training)
#         logits = tf.squeeze(logits)

#         self.variables = tf.contrib.framework.get_variables("resnet_v1_50/logits")
#         print('here',self.variables)
#         #### accuracy
#         predictions = tf.to_int32(tf.argmax(logits, 1))
#         labels = tf.to_int32(tf.argmax(input_y,1))
#         accuracy, tf_metric_update = tf.metrics.accuracy(labels, predictions, name="accuracy_metric")

#         #### cost
#         tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#         cost = tf.losses.get_total_loss()
#         return cost, accuracy
####### pretrained_resnet50


####### resnet50
layers = tf.keras.layers


class _IdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, data_format):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.conv2a = layers.Conv2D(
            filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')
        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')
        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 data_format,
                 strides=(2, 2)):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.conv2a = layers.Conv2D(
            filters1, (1, 1),
            strides=strides,
            name=conv_name_base + '2a',
            data_format=data_format)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')
        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b',
            data_format=data_format)
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')
        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')
        self.conv_shortcut = layers.Conv2D(
            filters3, (1, 1),
            strides=strides,
            name=conv_name_base + '1',
            data_format=data_format)
        self.bn_shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)
        x += shortcut
        return tf.nn.relu(x)


class resnet50():
    def __init__(self, data_format='channels_last', name='resnet50', trainable=True, include_top=True, pooling=None,
                 classes=1000):
        self.reuse = tf.AUTO_REUSE
        self.include_top = include_top
        self.name = name

        def conv_block(filters, stage, block, strides=(2, 2)):
            return _ConvBlock(
                3,
                filters,
                stage=stage,
                block=block,
                data_format=data_format,
                strides=strides)

        def id_block(filters, stage, block):
            return _IdentityBlock(
                3, filters, stage=stage, block=block, data_format=data_format)

        self.conv1 = layers.Conv2D(
            64, (7, 7),
            strides=(2, 2),
            data_format=data_format,
            padding='same',
            name='conv1')
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.max_pool = layers.MaxPooling2D(
            (3, 3), strides=(2, 2), data_format=data_format)

        # filters = [128, 128, 256, 512, 1024]
        # filters = [64, 64, 128, 256, 512]
        self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.l2b = id_block([64, 64, 256], stage=2, block='b')
        self.l2c = id_block([64, 64, 256], stage=2, block='c')
        self.l3a = conv_block([128, 128, 512], stage=3, block='a')
        self.l3b = id_block([128, 128, 512], stage=3, block='b')
        self.l3c = id_block([128, 128, 512], stage=3, block='c')
        self.l3d = id_block([128, 128, 512], stage=3, block='d')
        self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
        self.l4b = id_block([256, 256, 1024], stage=4, block='b')
        self.l4c = id_block([256, 256, 1024], stage=4, block='c')
        self.l4d = id_block([256, 256, 1024], stage=4, block='d')
        self.l4e = id_block([256, 256, 1024], stage=4, block='e')
        self.l4f = id_block([256, 256, 1024], stage=4, block='f')
        self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
        self.l5b = id_block([512, 512, 2048], stage=5, block='b')
        self.l5c = id_block([512, 512, 2048], stage=5, block='c')
        self.avg_pool = layers.AveragePooling2D(
            (7, 7), strides=(7, 7), data_format=data_format)
        if self.include_top:
            self.flatten = layers.Flatten()
            self.fc1000 = layers.Dense(classes, name='fc1000')
        else:
            reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
            reduction_indices = tf.constant(reduction_indices)
            if pooling == 'avg':
                self.global_pooling = functools.partial(
                    tf.reduce_mean,
                    reduction_indices=reduction_indices,
                    keep_dims=False)
            elif pooling == 'max':
                self.global_pooling = functools.partial(
                    tf.reduce_max, reduction_indices=reduction_indices, keep_dims=False)
            else:
                self.global_pooling = None

    def __call__(self, input_tensor_batch, input_y, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            input_tensor_batch_3_channel = tf.concat([input_tensor_batch, input_tensor_batch, input_tensor_batch],
                                                     axis=3)
            x = self.conv1(input_tensor_batch_3_channel)
            x = self.bn_conv1(x, training=training)
            x = tf.nn.relu(x)
            x = self.max_pool(x)
            x = self.l2a(x, training=training)
            x = self.l2b(x, training=training)
            x = self.l2c(x, training=training)
            x = self.l3a(x, training=training)
            x = self.l3b(x, training=training)
            x = self.l3c(x, training=training)
            x = self.l3d(x, training=training)
            x = self.l4a(x, training=training)
            x = self.l4b(x, training=training)
            x = self.l4c(x, training=training)
            x = self.l4d(x, training=training)
            x = self.l4e(x, training=training)
            x = self.l4f(x, training=training)
            x = self.l5a(x, training=training)
            x = self.l5b(x, training=training)
            x = self.l5c(x, training=training)
            # x = self.avg_pool(x)
            if self.include_top:
                logits = self.fc1000(self.flatten(x))
                cost = tf.losses.softmax_cross_entropy(onehot_labels=input_y, logits=logits)
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
                count_parameters(self.variables, name="classifier_parameter_num")
                return cost, accuracy
            elif self.global_pooling:
                return self.global_pooling(x)
            else:
                return x


####### resnet50


######resnet
BN_EPSILON = 0.001


def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension, training):
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride, training):
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel, training)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, training):
    in_channel = input_layer.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_layer, in_channel, training)
    relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, training, first_block=False):
    input_channel = input_layer.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, training)
    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, training)
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer
    output = conv2 + padded_input
    return output


class ResNet():
    def __init__(self, name='resnet_classifier'):
        self.reuse = tf.AUTO_REUSE
        self.name = name

    def __call__(self, input_tensor_batch, input_y, n, classes, training):
        layers = []
        filter_depth = 16
        #  filters = [16, 16, 32, 64]
        with tf.variable_scope(self.name, reuse=self.reuse):
            # [-1,w,h,3]
            with tf.variable_scope('conv0', reuse=self.reuse):
                if input_tensor_batch.get_shape()[3] == 3:
                    conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, filter_depth], 1, training)
                else:
                    conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 1, filter_depth], 1, training)
                activation_summary(conv0)
                layers.append(conv0)
            # [-1,w/2,h/2,features=filter_depth]

            for i in range(n):
                with tf.variable_scope('conv1_%d' % i, reuse=self.reuse):
                    if i == 0:
                        conv1 = residual_block(layers[-1], filter_depth, training, first_block=True)
                    else:
                        conv1 = residual_block(layers[-1], filter_depth, training)
                    activation_summary(conv1)
                    layers.append(conv1)
            # [-1,w/2,h/2,features=filter_depth]

            for i in range(n):
                with tf.variable_scope('conv2_%d' % i, reuse=self.reuse):
                    conv2 = residual_block(layers[-1], filter_depth * 2, training)
                    activation_summary(conv2)
                    layers.append(conv2)
            # [-1,w/4,h/4,features=filter_depth*2]

            for i in range(n):
                with tf.variable_scope('conv3_%d' % i, reuse=self.reuse):
                    conv3 = residual_block(layers[-1], filter_depth * 4, training)
                    layers.append(conv3)
                # assert conv3.get_shape().as_list()[1:] == [8, 8, 64]
            # [-1,w/8,h/8,features=filter_depth*4]

        with tf.variable_scope('fc', reuse=self.reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel, training)
            # relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(bn_layer, [1, 2])
            # assert global_pool.get_shape().as_list()[-1:] == [filter_depth*4]
            output = output_layer(global_pool, 1024)
            layers.append(output)
            logits = tf.layers.dense(layers[-1], units=classes)

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.conv_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.fc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc')

        self.variables = self.conv_variables + self.fc_variables
        count_parameters(self.variables, name="classifier_parameter_num")
        return cost, accuracy


######


#####
def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network


def Global_Average_Pooling(x, stride=1):
    return global_avg_pool(x, name='Global_avg_pooling')


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training),
                       lambda: batch_norm(inputs=x, is_training=training))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x, class_num):
    return tf.layers.dense(inputs=x, units=class_num)


class DenseNet():
    def __init__(self, nb_blocks, filters, class_num, training, dropout_rate, name='classifier'):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.reuse = tf.AUTO_REUSE
        self.name = name

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            in_channel = x.shape[-1]
            x = conv_layer(x, filter=int(in_channel) * 0.5, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)
            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
            x = Concatenation(layers_concat)
            return x

    def __call__(self, input_x, input_y):
        with tf.variable_scope(self.name, reuse=self.reuse):
            x = conv_layer(input_x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
            x = Max_Pooling(x, pool_size=[3, 3], stride=2)
            for i in range(self.nb_blocks):
                x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_' + str(i))
                x = self.transition_layer(x, scope='trans_' + str(i))
            x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')
            x = Batch_Normalization(x, training=self.training, scope='linear_batch')
            x = Relu(x)
            x = Global_Average_Pooling(x)
            x = flatten(x)
            logits = Linear(x, self.class_num)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        count_parameters(self.variables, name="classifier_parameter_num")
        return cost, accuracy


#############


def remove_duplicates(input_features):
    feature_name_set = set()
    non_duplicate_feature_set = []
    for feature in input_features:
        if feature.name not in feature_name_set:
            non_duplicate_feature_set.append(feature)
        feature_name_set.add(feature.name)
    return non_duplicate_feature_set


class densenet_architecture:
    def __init__(self, batch_size, layer_sizes, inner_layers, use_wide_connections=False, name="d"):
        self.reuse = tf.AUTO_REUSE
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.use_wide_connections = use_wide_connections
        self.build = True
        self.name = name

    def upscale(self, x, scale):
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]
        return tf.image.resize_nearest_neighbor(x, (h * scale, w * scale))

    def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None, transpose=False):
        self.conv_layer_num += 1
        if transpose:
            outputs = tf.layers.conv2d_transpose(inputs, num_filters, filter_size, strides=strides,
                                                 padding="SAME", activation=activation)
        elif not transpose:
            outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
                                       padding="SAME", activation=activation)
        return outputs

    def add_encoder_layer(self, input, name, training, layer_to_skip_connect, local_inner_layers, num_features,
                          dim_reduce=False, dropout_rate=0.0):
        [b1, h1, w1, d1] = input.get_shape().as_list()
        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()
            if h0 > h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect, int(layer_to_skip_connect.get_shape()[3]),
                                                     [3, 3], strides=(2, 2))
            else:
                skip_connect_layer = layer_to_skip_connect
        else:
            skip_connect_layer = layer_to_skip_connect
        current_layers = [input, skip_connect_layer]
        current_layers.extend(local_inner_layers)
        current_layers = remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)
        if dim_reduce:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(2, 2))
            outputs = leaky_relu(features=outputs)
            outputs = layer_norm(inputs=outputs, center=True, scale=True)
            # outputs = tf.nn.relu(outputs)
            # outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1))
            outputs = leaky_relu(features=outputs)
            outputs = layer_norm(inputs=outputs, center=True, scale=True)
            # outputs = tf.nn.relu(outputs)
            # outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
        return outputs

    def __call__(self, inputs, input_y, classes, training=False, dropout_rate=0.0):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.name, reuse=self.reuse):
            outputs = inputs
            encoder_layers = []
            current_layers = [outputs]
            with tf.variable_scope('conv_layers'):
                for i, layer_size in enumerate(self.layer_sizes):
                    encoder_inner_layers = [outputs]
                    with tf.variable_scope('g_conv{}'.format(i)):
                        if i == 0:
                            outputs = self.conv_layer(outputs, num_filters=64,
                                                      filter_size=(3, 3), strides=(2, 2))
                            outputs = leaky_relu(features=outputs)
                            outputs = layer_norm(inputs=outputs, center=True, scale=True)
                            # outputs = tf.nn.relu(outputs)
                            # outputs = tf.layers.batch_normalization(outputs, training=training, momentum=0.9)
                            current_layers.append(outputs)
                        else:
                            for j in range(self.inner_layers[i]):
                                outputs = self.add_encoder_layer(input=outputs,
                                                                 name="encoder_inner_conv_{}_{}"
                                                                 .format(i, j), training=training,
                                                                 layer_to_skip_connect=current_layers[-2],
                                                                 num_features=self.layer_sizes[i],
                                                                 dropout_rate=dropout_rate,
                                                                 dim_reduce=False,
                                                                 local_inner_layers=encoder_inner_layers)
                                current_layers.append(outputs)
                                encoder_inner_layers.append(outputs)
                            outputs = self.add_encoder_layer(input=outputs,
                                                             name="encoder_outer_conv_{}"
                                                             .format(i),
                                                             training=training,
                                                             layer_to_skip_connect=
                                                             current_layers[-2],
                                                             local_inner_layers=
                                                             encoder_inner_layers,
                                                             num_features=self.layer_sizes[i],
                                                             dropout_rate=dropout_rate,
                                                             dim_reduce=True)
                            current_layers.append(outputs)
                        encoder_layers.append(outputs)

            with tf.variable_scope('classifier_dense_block'):
                if self.use_wide_connections:
                    mean_encoder_layers = []
                    concat_encoder_layers = []
                    for layer in encoder_layers:
                        mean_encoder_layers.append(tf.reduce_mean(layer, axis=[1, 2]))
                        concat_encoder_layers.append(tf.layers.flatten(layer))
                    feature_level_flatten = tf.concat(mean_encoder_layers, axis=1)
                    location_level_flatten = tf.concat(concat_encoder_layers, axis=1)
                else:
                    feature_level_flatten = tf.reduce_mean(encoder_layers[-1], axis=[1, 2])
                    location_level_flatten = tf.layers.flatten(encoder_layers[-1])

                feature_level_dense = tf.layers.dense(feature_level_flatten, units=1024, activation=leaky_relu)
                # feature_level_dense = tf.layers.dense(feature_level_flatten, units=1024)
                combo_level_flatten = tf.concat([feature_level_dense, location_level_flatten], axis=1)

            with tf.variable_scope('classifier_out_block'):
                logits = tf.layers.dense(combo_level_flatten, units=classes)
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits))
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                # cost = tf.losses.softmax_cross_entropy(onehot_labels=input_y, logits=logits)
                # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                # cost = tf.losses.softmax_cross_entropy(onehot_labels=input_y, logits=logits)
                # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
                # # correct_prediction=tf.Print(correct_prediction,[correct_prediction],'correct_prediction')
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        if self.build:
            print("classification layers", self.conv_layer_num)
            count_parameters(self.variables, name="classifier_parameter_num")
        self.build = False
        return cost, accuracy


class densenet_classifier:
    def __init__(self, input_x_i, input_y, classes,
                 batch_size, layer_sizes, inner_layers, num_gpus,
                 use_wide_connections, is_training, augment, dropout_rate):
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.input_x_i = input_x_i
        self.input_y = input_y
        self.classes = classes
        self.is_training = is_training
        self.augment = augment
        self.dropout_rate = dropout_rate

        #### discriminator similar
        # self.classifier_network = densenet_architecture(batch_size=self.batch_size,
        #     layer_sizes=layer_sizes, inner_layers=inner_layers, use_wide_connections=use_wide_connections, name="classifier")

        ##### resnet
        # self.classifier_network = ResNet()

        ##### resnet50
        # self.classifier_network = resnet50(trainable=self.is_training,classes=self.classes)

        ##### usual densenet
        # self.classifier_network = DenseNet(nb_blocks=2, filters=4, class_num=self.classes, training=self.is_training, dropout_rate=self.dropout_rate)

        ##### mobilener
        # self.classifier_network = mobilenet_finetune()

        ##### pretrained_resnet
        # self.classifier_network = pretrained_resnet_architecture()

        ##### pretrained_resnet 18
        self.classifier_network = ResNet18(is_training)

    def rotate_data(self, image):
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image = tf.image.rot90(image, k=random_variable[0])
        return image

    def shifts_data(self, image):
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return image

    def add_gaussian_noise(self, image):
        noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
        noise_img = image + noise
        noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
        return noise_img

    def augment_rotate(self, image):
        r = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None))
        rotate_boolean = tf.equal(0, r, name="check-rotate-boolean")
        image = tf.cond(rotate_boolean[0], lambda: self.rotate_data(image),
                        lambda: image)
        image = tf.cond(rotate_boolean[0], lambda: self.shifts_data(image),
                        lambda: image)
        image = tf.cond(rotate_boolean[0], lambda: self.add_gaussian_noise(image),
                        lambda: image)
        return image

    def data_augment_batch(self, batch_images):
        images = tf.cond(self.augment, lambda: self.rotate_batch(batch_images),
                         lambda: batch_images)
        return images

    def rotate_batch(self, batch_images):
        shapes = map(int, list(batch_images.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked = tf.unstack(batch_images)
            new_images = []
            for image in batch_images_unpacked:
                rotate = self.augment_rotate(image)
                new_images.append(rotate)
            new_images = tf.stack(new_images)
            new_images = tf.reshape(new_images, (batch_size, x, y, c))
            return new_images

    def loss(self, gpu_id):
        with tf.name_scope("losses_{}".format(gpu_id)):
            input_x, input_y = self.input_x_i[gpu_id], self.input_y[gpu_id]
            input_x = tf.cond(self.is_training, lambda: self.data_augment_batch(input_x), lambda: input_x)
            ##### similar to discriminator
            # cost, accuracy = self.classifier_network(input_x, input_y, self.classes, training=self.is_training, dropout_rate=self.dropout_rate)

            ##### resnet
            # cost,accuracy = self.classifier_network(input_x, input_y, 2, self.classes,training=self.is_training)

            ##### resnet50
            # cost,accuracy = self.classifier_network(input_x,input_y,training=self.is_training)

            ##### usual densenet
            # cost,accuracy = self.classifier_network(input_x, input_y)

            ##### mobilenet
            # cost, accuracy = self.classifier_network(input_x,input_y,classes)

            ##### pretrained_resnet
            # cost,accuracy = self.classifier_network(input_x, input_y, self.classes,training=self.is_training)

            ##### pretrained_resnet_18
            cost, accuracy = self.classifier_network(input_x, input_y, self.classes)

            tf.add_to_collection('losses', cost)
            tf.add_to_collection('accuracy', accuracy)
            tf.summary.scalar('losses', cost)
            tf.summary.scalar('accuracy', accuracy)

        return {"losses": tf.add_n(tf.get_collection('losses'), name='total_loss')}, accuracy
        # return {
        # "losses":tf.add_n(tf.get_collection('losses'),name='total_loss'),
        # "accuracy":tf.add_n(tf.get_collection('accuracy'),name='total accuracy')
        # }

    def train(self, opts, losses):
        opt_ops = dict()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_ops["loss_opt_op"] = opts["loss_opt"].minimize(losses["losses"],
                                                               var_list=self.classifier_network.variables,
                                                               colocate_gradients_with_ops=True)
        return opt_ops

    def init_train(self, learning_rate=1e-3, beta1=0.9, beta2=0.99):
        losses = dict()
        accuracies = dict()
        opts = dict()

        if self.num_gpus > 0:
            device_ids = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]
        else:
            device_ids = ['/cpu:0']

        for gpu_id, device_id in enumerate(device_ids):
            with tf.device(device_id):
                total_losses, accuracy = self.loss(gpu_id=gpu_id)
                for key, value in total_losses.items():
                    if key not in losses.keys():
                        losses[key] = [value]
                    else:
                        losses[key].append(value)

        for key in list(losses.keys()):
            losses[key] = tf.reduce_mean(losses[key], axis=0)
            opts[key.replace("losses", "loss_opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                             learning_rate=learning_rate)

            # opts[key.replace("losses", "loss_opt")] = tf.train.MomentumOptimizer(0.001, 0.9)

        summary = tf.summary.merge_all()
        apply_grads_ops = self.train(opts=opts, losses=losses)
        accuracy = tf.reduce_mean(accuracy)

        return summary, losses, accuracy, apply_grads_ops














