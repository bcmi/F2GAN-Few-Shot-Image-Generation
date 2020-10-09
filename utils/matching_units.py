import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool, leaky_relu
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu
from utils.network_summary import count_parameters

from utils.sn import spectral_normed_weight, spectral_norm


class g_embedding_bidirectionalLSTM:
    def __init__(self, name, layer_sizes, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.name = name

    def __call__(self, inputs, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope("encoder"):

                fw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                         for i in range(len(self.layer_sizes))]
                bw_lstm_cells_encoder = [rnn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
                                         for i in range(len(self.layer_sizes))]

                outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
                    fw_lstm_cells_encoder,
                    bw_lstm_cells_encoder,
                    inputs,
                    dtype=tf.float32
                )

            # print("g out shape", tf.stack(outputs, axis=1).get_shape().as_list())

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs

class f_embedding_bidirectionalLSTM:
    def __init__(self, name, layer_size, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.name = name

    def __call__(self, support_set_embeddings, target_set_embeddings, K, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        b, k, h_g_dim = support_set_embeddings.get_shape().as_list()
        b, h_f_dim = target_set_embeddings.get_shape().as_list()
        with tf.variable_scope(self.name, reuse=self.reuse):
            fw_lstm_cells_encoder = rnn.LSTMCell(num_units=self.layer_size, activation=tf.nn.tanh)
            attentional_softmax = tf.ones(shape=(b, k)) * (1.0/k)
            h = tf.zeros(shape=(b, h_g_dim))
            c_h = (h, h)
            c_h = (c_h[0], c_h[1] + target_set_embeddings)
            for i in range(K):
                attentional_softmax = tf.expand_dims(attentional_softmax, axis=2)
                attented_features = support_set_embeddings * attentional_softmax
                attented_features_summed = tf.reduce_sum(attented_features, axis=1)
                c_h = (c_h[0], c_h[1] + attented_features_summed)
                print('1',target_set_embeddings)
                x, h_c = fw_lstm_cells_encoder(inputs=target_set_embeddings, state=c_h)
                attentional_softmax = tf.layers.dense(x, units=k, activation=tf.nn.softmax, reuse=self.reuse)
                self.reuse = True

        outputs = x
        # print("out shape", tf.stack(outputs, axis=0).get_shape().as_list())
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # print(self.variables)
        return outputs



# class DistanceNetwork:
#     def __init__(self):
#         self.reuse = False

#     def __call__(self, support_set, input_image, name, training=False):
#         with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
#             eps = 1e-10
#             similarities = []
#             for support_image in tf.unstack(support_set, axis=0):
#                 sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)
#                 sum_target = tf.reduce_sum(tf.square(input_image), 1, keep_dims=True)
#                 support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))
#                 target_magnitude = tf.rsqrt(tf.clip_by_value(sum_target, eps, float("inf")))
#                 dot_product = tf.matmul(tf.expand_dims(input_image, 1), tf.expand_dims(support_image, 2))
#                 dot_product = tf.squeeze(dot_product, [1, ])
#                 cosine_similarity = dot_product * support_magnitude * target_magnitude
#                 similarities.append(cosine_similarity)

#         similarities = tf.concat(axis=1, values=similarities)
#         # similarities = tf.nn.softmax(similarities)
#         self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
#         return similarities


class DistanceNetwork:
    def __init__(self):
        self.reuse = False

    def __call__(self, support_set, input_image, name, training=False):
        with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
            eps = 1e-10
            similarities = []
            similarities_standard = []
            for support_image in tf.unstack(support_set, axis=0):
                sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)
                support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))
                sum_target = tf.reduce_sum(tf.square(input_image), 1, keep_dims=True)
                target_magnitude = tf.rsqrt(tf.clip_by_value(sum_target, eps, float("inf")))
                dot_product = tf.matmul(tf.expand_dims(input_image, 1), tf.expand_dims(support_image, 2))
                dot_product = tf.squeeze(dot_product, [1, ])
                cosine_similarity = dot_product * support_magnitude
                cosine_similarity_standard = cosine_similarity * target_magnitude
                similarities.append(cosine_similarity)
                similarities_standard.append(cosine_similarity_standard)

            
        
        similarities = tf.concat(axis=1, values=similarities)
        mean_magnitude = tf.reduce_mean(target_magnitude)
        similarities_standard = tf.concat(axis=1,values=similarities_standard)


        minvalue = tf.reduce_min(similarities_standard,axis=1)
        maxvalue = tf.reduce_max(similarities_standard,axis=1)
        meanvalue, variance = tf.nn.moments(similarities_standard,axes=1)
        minvalue  = tf.expand_dims(minvalue ,axis=1)
        maxvalue=tf.expand_dims(maxvalue,axis=1)
        meanvalue=tf.expand_dims(meanvalue,axis=1)
        variance=tf.expand_dims(variance,axis=1)
       



        all_data = tf.concat([similarities_standard,target_magnitude,minvalue,maxvalue,meanvalue,variance],axis=1)
        scale = tf.random_uniform(shape=[int(similarities.get_shape()[0]),1],minval=1,maxval=10)  
        scale = tf.concat([scale,scale,scale],axis=1) 
        similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = tf.nn.softmax(tf.multiply(scale,similarities))
        # similarities_standard = tf.nn.softmax(1e1*similarities)
        # similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = tf.nn.softmax(similarities)
        # similarities_standard = similarities
        # similarities_standard = tf.snn.softmax(similarities)
        # similarities_standard_scale = tf.random_uniform(shape=[1],minval=1,maxval=100)
        # similarities_standard = similarities_standard_scale : * similarities_standard
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
        return similarities, similarities_standard, all_data


# class DistanceNetwork:
#     def __init__(self):
#         self.reuse = False

#     def __call__(self, support_set, input_image, name, training=False):
#         with tf.name_scope('distance-module' + name), tf.variable_scope('distance-module', reuse=self.reuse):
#             eps = 1e-10
#             similarities = []
#             for support_image in tf.unstack(support_set, axis=0)::
#                 distance_current = tf.sqrt(tf.reduce_sum(tf.square(support_image-input_image), 1, keep_dims=True))
#                 similarities.append(distance_current)
#         similarities = tf.concat(axis=1, values=similarities)        
#         sum_value=tf.reduce_sum(similarities,axis=1)
#         similarities_value = []
#         for i in range(similarities.get_shape()[0]):
#             similarities_value.append(similarities[i]/sum_value[i])
#         similarities = tf.stack(similarities_value,axis=0)
#         similarities_standard = tf.nn.softmax(similarities)
#         self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='distance-module')
#         return similarities, similarities_standard

class AttentionalClassify:
    def __init__(self):
        self.reuse = False

    def __call__(self, similarities, support_set_y, name, training=False):
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification',
                                                                                   reuse=self.reuse):
            preds = tf.squeeze(tf.matmul(tf.expand_dims(similarities, 1), support_set_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds




class Classifier:
    def __init__(self, name, batch_size, layer_sizes, num_channels=1):
        """
        Builds a CNN to produce embeddings
        :param batch_size: Batch size for experiment
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        self.reuse = tf.AUTO_REUSE
        self.name = name
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        assert len(self.layer_sizes) == 4, "layer_sizes should be a list of length 4"

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            outputs = image_input
            with tf.variable_scope('conv_layers'):
                for idx, num_filters in enumerate(self.layer_sizes):
                    with tf.variable_scope('g_conv_{}'.format(idx)):
                        if idx == len(self.layer_sizes) - 1:
                            outputs = tf.layers.conv2d(outputs, num_filters, [2, 2], strides=(1, 1),
                                                       padding='VALID')
                        else:
                            outputs = tf.layers.conv2d(outputs, num_filters, [3, 3], strides=(1, 1),
                                                               padding='VALID')
                        outputs = leaky_relu(outputs)
                        outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None,
                                                                       decay=0.99,
                                                                       scale=True, center=True,
                                                                       is_training=training)
                        outputs = max_pool(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME')
                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            image_embedding = tf.contrib.layers.flatten(outputs)


        self.reuse = tf.AUTO_REUSE
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return image_embedding




weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope=None):
    with tf.variable_scope(scope):
        # if pad_type == 'zero' :
        #     x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        # if pad_type == 'reflect' :
        #     x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='SAME', name='conv_sn')
            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias, padding="SAME", name='conv')
        return x


# def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None,
#                transpose=False, w_size=None, h_size=None,scope='scope_0'):
#     self.conv_layer_num += 1
#     if transpose:
#         outputs = self.upscale(inputs, h_size=h_size, w_size=w_size)
#         outputs = tf.layers.conv2d_transpose(outputs, num_filters, filter_size,
#                                              strides=strides,
#                                    padding="SAME", activation=activation)
#     elif not transpose:
#         outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
#                                              padding="SAME", activation=activation)
#     return outputs


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope=None):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_normed_weight(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding='SAME', name='deconv_sn')

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias, name='deconv')

        return x


def fully_conneted(x, units, use_bias=True, sn=False):
    x = tf.layers.flatten(x)
    shape = x.get_shape().as_list()
    channels = shape[-1]

    if sn:
        w = tf.get_variable("kernel", [channels, units], tf.float32,
                            initializer=weight_init, regularizer=weight_regularizer)
        if use_bias:
            bias = tf.get_variable("bias", [units],
                                   initializer=tf.constant_initializer(0.0))

            x = tf.matmul(x, spectral_norm(w)) + bias
        else:
            x = tf.matmul(x, spectral_norm(w))

    else:
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                            use_bias=use_bias)

    return x


def max_pooling(x, kernel=2, stride=2):
    return tf.layers.max_pooling2d(x, pool_size=kernel, strides=stride)


def avg_pooling(x, kernel=2, stride=2):
    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride)


def global_avg_pooling(x):
    """
    Incoming Tensor shape must be 4-D
    """
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap


def flatten(x):
    return tf.layers.flatten(x)


def lrelu(x, alpha=0.2):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def swish(x):
    return x * sigmoid(x)


def remove_duplicates(input_features):
    """
    Remove duplicate entries from layer list.
    :param input_features: A list of layers
    :return: Returns a list of unique feature tensors (i.e. no duplication).
    """
    feature_name_set = set()
    non_duplicate_feature_set = []
    for feature in input_features:
        if feature.name not in feature_name_set:
            non_duplicate_feature_set.append(feature)
        feature_name_set.add(feature.name)
    return non_duplicate_feature_set






class Unet_encoder:
    def __init__(self,  layer_sizes,inner_layers):
        self.reuse = tf.AUTO_REUSE
        self.layer_sizes = layer_sizes
        self.inner_layers = inner_layers

    def conv_layer(self, inputs, num_filters, filter_size, strides, scope, activation=None,
                   transpose=False, w_size=None, h_size=None, sn=True):
        if transpose:
            outputs = self.upscale(inputs, h_size=h_size, w_size=w_size)
            outputs = deconv(outputs, channels=num_filters, kernel=filter_size[0], stride=strides[0], use_bias=True,
                             sn=sn, scope=scope)
        elif not transpose:
            outputs = conv(inputs, channels=num_filters, kernel=filter_size[0], stride=strides[0], pad=2, sn=sn,
                           scope=scope)
        return outputs

    def add_encoder_layer(self, input, name, training, dropout_rate, layer_to_skip_connect, local_inner_layers,
                          num_features, dim_reduce=False, scope=None):

        """
        Adds a resnet encoder layer.
        :param input: The input to the encoder layer
        :param training: Flag for training or validation
        :param dropout_rate: A float or a placeholder for the dropout rate
        :param layer_to_skip_connect: Layer to skip-connect this layer to
        :param local_inner_layers: A list with the inner layers of the current Multi-Layer
        :param num_features: Number of feature maps for the convolutions
        :param dim_reduce: Boolean value indicating if this is a dimensionality reducing layer or not
        :return: The output of the encoder layer
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()

        if len(layer_to_skip_connect) >= 2:
            layer_to_skip_connect = layer_to_skip_connect[-2]
        else:
            layer_to_skip_connect = None

        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()
            if h0 > h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect, int(layer_to_skip_connect.get_shape()[3]),
                                                     [3, 3], strides=(2, 2), scope='scope1')
            else:
                skip_connect_layer = layer_to_skip_connect
            # print('1',input)
            # print('2',skip_connect_layer)
            current_layers = [input, skip_connect_layer]
        else:
            current_layers = [input]

        current_layers.extend(local_inner_layers)
        current_layers = remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)

        if dim_reduce:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(2, 2), scope='scope2')
            outputs = leaky_relu(outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=False, scope='norm_en')
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1), scope='scope2')
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=False, scope='norm_en')

        return outputs

    def __call__(self, image_input, scope, training=False, dropout_rate=0.0):
        outputs = image_input
        encoder_layers = []
        current_layers = [outputs]
        with tf.variable_scope(scope):
            for i, layer_size in enumerate(self.layer_sizes):
                encoder_inner_layers = [outputs]
                with tf.variable_scope('g_conv{}'.format(i)):
                    if i == 0:  
                        outputs = self.conv_layer(outputs, num_filters=64,
                                                  filter_size=(3, 3), strides=(2, 2),
                                                  scope='g_conv{}'.format(i))
                        outputs = leaky_relu(features=outputs)
                        outputs = batch_norm(outputs, decay=0.99, scale=True,
                                             center=True, is_training=training,
                                             renorm=True, scope='bn_1')
                        current_layers.append(outputs)
                        encoder_inner_layers.append(outputs)
                    else:
                        for j in range(self.inner_layers[i]):  # Build the inner Layers of the MultiLayer
                            with tf.variable_scope('g_conv_inner_layer{}'.format(j)):
                                outputs = self.add_encoder_layer(input=outputs,
                                                                 training=training,
                                                                 name="encoder_layer_{}_{}".format(i,
                                                                                                   j),
                                                                 layer_to_skip_connect=current_layers,
                                                                 num_features=self.layer_sizes[i],
                                                                 dim_reduce=False,
                                                                 local_inner_layers=encoder_inner_layers,
                                                                 dropout_rate=dropout_rate,
                                                                 scope="encoder_layer_{}_{}".format(i,
                                                                                                    j))
                                encoder_inner_layers.append(outputs)
                                # current_layers.append(outputs)
                        # add final dim reducing conv layer for this MultiLayer
                        outputs = self.add_encoder_layer(input=outputs,
                                                         name="encoder_layer_{}".format(j),
                                                         training=training,
                                                         layer_to_skip_connect=current_layers,
                                                         local_inner_layers=encoder_inner_layers,
                                                         num_features=self.layer_sizes[i],
                                                         dim_reduce=True, dropout_rate=dropout_rate,
                                                         scope="encoder_layer_{}".format(i))
                        current_layers.append(outputs)
                    encoder_layers.append(outputs)
                    # print('{}_th encoder output', outputs)

        return outputs, encoder_layers



import logging
import math

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from PIL import Image, ImageDraw


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])

    
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_bilinear)
    return y, flow


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def image2edge(image):
    """Convert image to edges.
    """
    out = []
    for i in range(image.shape[0]):
        img = cv2.Laplacian(image[i, :, :, :], cv2.CV_64F, ksize=3, scale=2)
        out.append(img)
    return np.float32(np.uint8(out))
