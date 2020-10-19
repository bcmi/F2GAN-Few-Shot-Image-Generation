import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu
from utils.network_summary import count_parameters
from utils.matching_units import g_embedding_bidirectionalLSTM, f_embedding_bidirectionalLSTM, DistanceNetwork, \
    AttentionalClassify, Classifier, Unet_encoder, contextual_attention
from utils.sn import spectral_normed_weight, spectral_norm
import numpy as np
from tensorflow.contrib import layers
import matplotlib
import matplotlib.cm

slim = tf.contrib.slim

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


##### visualization attention map
def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
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
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1] + [3])
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
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1] + [3])
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


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise




def random_choice(inputs, n_samples):
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)
    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)
    return tf.gather(inputs, ind, name="random_choice")




def gaussian_noise_change_layer(x, sim):
    noise = tf.random_normal(shape=[int(x.get_shape()[0]), int(x.get_shape()[1]), int(x.get_shape()[2])], mean=0.0,
                             stddev=1, dtype=tf.float32)
    condition = tf.random_uniform(shape=x.get_shape(), minval=0, maxval=1) < sim  # retile if needed
    x_image = tf.where(condition, x, noise)
    return x_image


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope=None):
    with tf.variable_scope(scope):
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


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


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


def sn_non_local_block_sim(x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        batch_size, h, w, num_channels = x.get_shape().as_list()
        location_num = h * w
        downsampled_num = location_num // 4

        # theta path
        theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
        theta = tf.reshape(
            theta, [batch_size, location_num, num_channels // 8])

        # phi path
        phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(
            phi, [batch_size, downsampled_num, num_channels // 8])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        # g path
        g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(
            g, [batch_size, downsampled_num, num_channels // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
        sigma = tf.get_variable(
            'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
        return x + sigma * attn_g


class UResNetGenerator:
    def __init__(self, layer_sizes, layer_padding, batch_size, num_channels=1,
                 inner_layers=0, name="g", matching=0, fce=0, full_context_unroll_k=4, average_per_class_embeddings=0):
        """
        Initialize a UResNet generator.
        :param layer_sizes: A list with the filter sizes for each MultiLayer e.g. [64, 64, 128, 128]
        :param layer_padding: A list with the padding type for each layer e.g. ["SAME", "SAME", "SAME", "SAME"]
        :param batch_size: An integer indicating the batch size
        :param num_channels: An integer indicating the number of input channels
        :param inner_layers: An integer indicating the number of inner layers per MultiLayer
        """
        self.training = True
        self.print = False
        self.reuse = tf.AUTO_REUSE
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        self.layer_padding = layer_padding
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.build = True
        self.matching = matching
        self.name = name
        self.fce = fce
        self.average_per_class_embeddings = average_per_class_embeddings
        self.full_context_K = full_context_unroll_k
        self.Matching_classifier = Classifier(name="matching_classifier_net", batch_size=self.batch_size,
                                              num_channels=num_channels, layer_sizes=[64, 64, 64, 64])

        self.g_lstm = g_embedding_bidirectionalLSTM(name="g_lstm", layer_sizes=[512], batch_size=self.batch_size)
        self.f_lstm = f_embedding_bidirectionalLSTM(name="f_attlstm", layer_size=1024, batch_size=self.batch_size)

        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.unet_encoder = Unet_encoder(layer_sizes=self.layer_sizes, inner_layers=self.inner_layers)

    def upscale(self, x, h_size, w_size):
        """
        Upscales an image using nearest neighbour
        :param x: Input image
        :param h_size: Image height size
        :param w_size: Image width size
        :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h_size, w_size))

    def conv_layer(self, inputs, num_filters, filter_size, strides, scope, activation=None,
                   transpose=False, w_size=None, h_size=None, sn=True):
        self.conv_layer_num += 1
        if transpose:
            outputs = self.upscale(inputs, h_size=h_size, w_size=w_size)
            outputs = deconv(outputs, channels=num_filters, kernel=filter_size[0], stride=strides[0], use_bias=True,
                             sn=sn, scope=scope)
        elif not transpose:
            outputs = conv(inputs, channels=num_filters, kernel=filter_size[0], stride=strides[0], pad=2, sn=sn,
                           scope=scope)
        return outputs

    def attention(self, ff, bf, channels, sn, scope='attention'):
        with tf.variable_scope(scope):
            f = conv(bf, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
            g = conv(ff, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
            h = conv(ff, channels, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=ff.shape)  # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=sn, scope='attn_conv')

            x = gamma * o + ff

        return x, beta

    def google_attention(self, ff, bf, channels, scope='attention'):
        with tf.variable_scope(scope):
            batch_size, height, width, num_channels = x.get_shape().as_list()
            f = conv(ff, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            f = max_pooling(f)

            g = conv(bf, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']

            h = conv(ff, channels // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            h = max_pooling(h)

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            x = gamma * o + x

        return x

    def resize_batch(self, batch_images, size):

        """
        Resize image batch using nearest neighbour
        :param batch_images: Image batch
        :param size: Size to upscale to
        :return: Resized image batch.
        """
        images = tf.image.resize_images(batch_images, size=size, method=ResizeMethod.NEAREST_NEIGHBOR)

        return images

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

    def add_decoder_layer(self, input, name, training, dropout_rate, layer_to_skip_connect, local_inner_layers,
                          num_features, dim_upscale=False, h_size=None, w_size=None, scope=None):

        """
        Adds a resnet decoder layer.
        :param input: Input features
        :param name: Layer Name
        :param training: Training placeholder or boolean flag
        :param dropout_rate: Float placeholder or float indicating the dropout rate
        :param layer_to_skip_connect: Layer to skip connect to.
        :param local_inner_layers: A list with the inner layers of the current MultiLayer
        :param num_features: Num feature maps for convolution
        :param dim_upscale: Dimensionality upscale
        :param h_size: Height to upscale to
        :param w_size: Width to upscale to
        :return: The output of the decoder layer
    """
        [b1, h1, w1, d1] = input.get_shape().as_list()
        if len(layer_to_skip_connect) >= 2:
            layer_to_skip_connect = layer_to_skip_connect[-2]
        else:
            layer_to_skip_connect = None

        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()

            if h0 < h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect,
                                                     int(layer_to_skip_connect.get_shape()[3]),
                                                     [3, 3], strides=(1, 1),
                                                     transpose=True,
                                                     h_size=h_size,
                                                     w_size=w_size, scope='scope1')
            else:
                skip_connect_layer = layer_to_skip_connect
            current_layers = [input, skip_connect_layer]
        else:
            current_layers = [input]

        current_layers.extend(local_inner_layers)
        current_layers = remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)

        if dim_upscale:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1),
                                      transpose=True, w_size=w_size, h_size=h_size, scope='scope2')
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs,
                                 decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True, scope='norm_de')
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training, name='drop_1')
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1),
                                      transpose=False, scope='scope2')
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True, scope='norm_de')

        return outputs

    ########### KL loss calculation ############
    def encoder(self, input_tensor, output_size):
        net = input_tensor
        net = layers.conv2d(net, 32, 5, stride=2)
        net = layers.conv2d(net, 64, 5, stride=2)
        net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
        net = layers.dropout(net, keep_prob=0.9)
        net = layers.flatten(net)
        return layers.fully_connected(net, output_size, activation_fn=None)

    def VAE_kl_loss(self, input_tensor, hidden_size, training, dropout, scope):
        with tf.variable_scope("model") as scope:
            # encoded = self.encoder(input_tensor, hidden_size * 2)
            encoded, _ = self.unet_encoder(image_input=input_tensor, training=training, dropout_rate=dropout,
                                           scope=scope)
            encoded_mean_std = layers.flatten(encoded)
            encoded_mean_std = layers.fully_connected(encoded_mean_std, hidden_size * 2, activation_fn=None)
            mean = encoded_mean_std[:, :hidden_size]
            stddev = tf.sqrt(tf.exp(encoded_mean_std[:, hidden_size:]))
            epsilon = tf.random_normal([tf.shape(mean)[0], hidden_size])

            if self.print:
                mean = tf.Print(mean, [mean], 'mean')
                stddev = tf.Print(stddev, [stddev], 'stddev')
            z = mean + epsilon * stddev
            kl_loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1)
        return encoded, z, kl_loss

    ########## reconstruction loss, l1 loss ###########
    def reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
        l1_loss = tf.reduce_mean(tf.abs(output_tensor - target_tensor)) + epsilon
        return l1_loss

    ######## reconstruction loss, l2 loss ##########
    def reconstruction_cost_l2(self, output_tensor, target_tensor, epsilon=1e-8):
        x = tf.reduce_mean(tf.square(output_tensor - target_tensor)) + epsilon
        return x

    def feature_matching_cost(self, output_tensor, target_tensor, epsilon=1e-8):
        x = tf.reduce_mean(tf.square(output_tensor - target_tensor)) + epsilon
        return x

    def crossentropy_softmax(self, outputs, targets):
        normOutputs = outputs - tf.reduce_max(outputs, axis=-1)[:, None]
        logProb = normOutputs - tf.log(tf.reduce_sum(tf.exp(normOutputs), axis=-1)[:, None])
        return tf.reduce_mean(tf.reduce_sum(targets * logProb, axis=1))
        # return tf.reduce_sum(targets * logProb, axis=1)

    def attention_module(self, attention_map, attention_target, name, training):
        outputs_for_attention = tf.concat([attention_map, attention_target], axis=3)
        ##### g
        outputs_attention = conv(outputs_for_attention, channels=128, kernel=3, stride=1,
                                 pad=1, sn=False, scope='conv1_{}'.format(name))
        outputs_attention = tf.nn.relu(outputs_attention)
        outputs_attention = batch_norm(outputs_attention,
                                       decay=0.99, scale=True,
                                       center=True, is_training=training,
                                       renorm=True, scope='bn1_{}'.format(name))

        outputs_attention = conv(outputs_attention, channels=64, kernel=3, stride=1, pad=1,
                                 sn=False, scope='conv2_{}'.format(name))
        outputs_attention = tf.nn.relu(outputs_attention)
        outputs_attention = batch_norm(outputs_attention,
                                       decay=0.99, scale=True,
                                       center=True, is_training=training,
                                       renorm=True, scope='bn2_{}'.format(name))

        outputs_attention = conv(outputs_attention, channels=1, kernel=1, stride=1, pad=0,
                                 sn=False, scope='conv3_{}'.format(name))
        outputs_attention = tf.nn.sigmoid(outputs_attention)
        # print('attention',outputs_attention) (25, 11, 11, 1),

        ##### f
        outputs_channel = tf.reduce_mean(outputs_for_attention, axis=[1, 2])
        outputs_channel = tf.layers.dense(outputs_channel,
                                          units=attention_target.get_shape()[-1],
                                          activation=tf.nn.relu,
                                          name='dense1_{}'.format(name))
        outputs_channel = batch_norm(outputs_channel,
                                     decay=0.99, scale=True,
                                     center=True, is_training=training,
                                     renorm=True, scope='bn3_{}'.format(name))

        outputs_channel = tf.layers.dense(outputs_channel,
                                          units=attention_target.get_shape()[-1],
                                          name='dense2_{}'.format(name))
        outputs_channel = tf.nn.sigmoid(outputs_channel)

        ##### g * f
        outputs_channel = tf.expand_dims(outputs_channel, axis=1)
        outputs_channel = tf.expand_dims(outputs_channel, axis=1)
        attention = tf.multiply(outputs_attention, outputs_channel)
        outpus = tf.multiply(attention, attention_target)
        return outpus, attention

    def bi_attention_module(self, source, target, name, training):
        target_attention, attention_map_target = self.attention_module(source, target, '{}_1'.format(name), training)
        source_attention, attention_map_source = self.attention_module(target, source, '{}_2'.format(name), training)
        outputs = tf.concat([target_attention, source_attention], axis=3)
        return outputs

    def __call__(self, z_inputs, z_inputs_2, input_batch_a, input_support_b, y_batch_a, y_support_b, selected_classes,
                 support_num, is_z2, is_z2_vae, z_dim, training=False, dropout_rate=0.0, z1z2_training=True):
        z1_random = True
        sim_random = True
        Z_dim = int(z_inputs.get_shape()[1])
        if self.matching > 0:
            conditional_input = input_batch_a
            support_input = input_support_b

            with tf.variable_scope(self.name, reuse=self.reuse):
                with tf.variable_scope('KL_layer'):
                    if not z1_random:
                        z1_feature, z_inputs_embedding, KL_loss_1 = self.VAE_kl_loss(conditional_input, z_dim, training,
                                                                                     dropout_rate, 'unet_B')
                        z_inputs = tf.cond(z1z2_training, lambda: z_inputs_embedding,
                                           lambda: z_inputs)
                        if is_z2_vae > 0:
                            z2_feature, z_inputs_2_embedding, KL_loss_2 = self.VAE_kl_loss(conditional_input, z_dim,
                                                                                           training, dropout_rate,
                                                                                           'unet_B')
                            z_inputs_2 = tf.cond(z1z2_training, lambda: z_inputs_2_embedding,
                                                 lambda: z_inputs_2)
                            KL_loss = KL_loss_2 + KL_loss_1

                        else:
                            z_inputs_2 = z_inputs_2
                            KL_loss = KL_loss_1


                    else:
                        z_inputs = z_inputs
                        if is_z2_vae > 0:
                            z_inputs_2_embedding, KL_loss_2 = self.VAE_kl_loss(conditional_input, z_dim)
                            z_inputs_2 = tf.cond(z1z2_training, lambda: z_inputs_2_embedding,
                                                 lambda: z_inputs_2)
                            KL_loss = KL_loss_2

                        else:
                            z_inputs_2 = z_inputs_2
                            KL_loss = tf.zeros([self.batch_size])

                    if self.print:
                        KL_loss = tf.Print(KL_loss, [KL_loss], 'loss_KL')
                    KL_loss = KL_loss

                if not sim_random:
                    matching_feature = tf.stack(matching_feature, axis=0)
                    similarities, similarities_standard, similarities_data = self.dn(support_set=matching_feature,
                                                                                     input_image=f_encoded_z_input,
                                                                                     training=training, name='dn')
                    similarities = similarities_standard
                else:
                    similarity_total = []
                    for i in range(support_num):
                        similarity_total.append(tf.random_uniform(shape=[self.batch_size, 1], minval=1, maxval=10))
                    similarity_total = tf.concat(similarity_total, axis=1)
                    similarity_sum_total = tf.expand_dims(tf.reduce_sum(similarity_total, axis=1), axis=1)
                    similarity_sum = tf.tile(similarity_sum_total, [1, support_num])
                    similarities = tf.divide(similarity_total, similarity_sum)

                ######## with tf.variable_scope('prototype_layer'):
                conditional_input_expand = tf.expand_dims(conditional_input, axis=1)
                batch_support = tf.concat([support_input, conditional_input_expand], axis=1)
                matching_feature = []
                prototype_feature = []
                encoder_layers_whole = []
                for k, image in enumerate(tf.unstack(batch_support, axis=1)):
                    if k < support_num:
                        current_image = []
                        for n in range(image.get_shape()[0]):
                            current_image.append(gaussian_noise_change_layer(image[n], similarities[n, k]))
                        image = tf.stack(current_image, axis=0)
                        g_conv_encoder, encoder_layers = self.unet_encoder(image_input=image, training=training,
                                                                           dropout_rate=dropout_rate,
                                                                           scope='unet')
                        prototype_feature.append(g_conv_encoder)
                        encoder_layers_whole.append(encoder_layers)

                        support_set_cnn_embed = g_conv_encoder
                        support_set_cnn_embed_reshape = tf.reshape(support_set_cnn_embed,
                                                                   [support_set_cnn_embed.get_shape()[0],
                                                                    support_set_cnn_embed.get_shape()[1] *
                                                                    support_set_cnn_embed.get_shape()[2] *
                                                                    support_set_cnn_embed.get_shape()[3]])
                        matching_feature.append(support_set_cnn_embed_reshape)

                    else:
                        g_conv_encoder, encoder_layers = self.unet_encoder(image_input=image, training=training,
                                                                           dropout_rate=dropout_rate,
                                                                           scope='unet_B')
                        xb_reconstruction = g_conv_encoder
                        xb_encode = encoder_layers

                with tf.variable_scope('matching_layer'):
                    similarities = tf.stop_gradient(similarities)
                    # tf.summary.text('similarity', tf.as_string(similarities[:5]))
                    # similarities = tf.Print(similarities, [similarities], 'similarities')

                ############### make preparation for the prototype processing, multiply the attention
                print('similarities', similarities)
                with tf.variable_scope('matching_prototype'):
                    prototype_feature = tf.stack(prototype_feature, axis=1)
                    prototype_feature_reshape = tf.reshape(prototype_feature, [prototype_feature.get_shape()[0],
                                                                               prototype_feature.get_shape()[1],
                                                                               prototype_feature.get_shape()[2] *
                                                                               prototype_feature.get_shape()[3] *
                                                                               prototype_feature.get_shape()[4]])

                    aggregated_feature_reshape = self.classify(similarities,
                                                               support_set_y=prototype_feature_reshape,
                                                               training=training, name='classify')
                    aggregated_feature = tf.reshape(aggregated_feature_reshape,
                                                    [prototype_feature.get_shape()[0], prototype_feature.get_shape()[2],
                                                     prototype_feature.get_shape()[3],
                                                     prototype_feature.get_shape()[4]])
                   

                ################ encoder layer with attention########################
                with tf.variable_scope('encoder_layer_attention'):
                    encoder_layers_attention = []
                    for m in range(len(encoder_layers_whole[0])):
                        current_layer = []
                        for n in range(len(encoder_layers_whole)):
                            current_layer.append(encoder_layers_whole[n][m])

                        current_layer = tf.stack(current_layer, axis=1)
                        current_layer_reshape = tf.reshape(current_layer,
                                                           [current_layer.get_shape()[0], current_layer.get_shape()[1],
                                                            current_layer.get_shape()[2] * current_layer.get_shape()[
                                                                3] * current_layer.get_shape()[4]])

                        current_layer_attention = self.classify(similarities,
                                                                support_set_y=current_layer_reshape, training=training,
                                                                name='classify')
                        current_layer_attention = tf.reshape(current_layer_attention, [current_layer.get_shape()[0],
                                                                                       current_layer.get_shape()[2],
                                                                                       current_layer.get_shape()[3],
                                                                                       current_layer.get_shape()[4]])
                        encoder_layers_attention.append(current_layer_attention)
                    encoder_layers = encoder_layers_attention

                ############concated with g_encode and random z############
                with tf.variable_scope("vector_expansion"):  # Used for expanding the z injected noise to match the
                    num_filters = 8
                    z_layers = []
                    concat_shape = [layer_shape.get_shape().as_list() for layer_shape in encoder_layers]
                    for i in range(len(self.inner_layers)):
                        h = concat_shape[len(encoder_layers) - 1 - i][1]
                        w = concat_shape[len(encoder_layers) - 1 - i][1]

                        z_dense = tf.layers.dense(z_inputs_2, h * w * num_filters)
                        z_reshape_noise = tf.reshape(z_dense, [np.shape(aggregated_feature)[0], h, w, num_filters])
                        num_filters /= 2
                        num_filters = int(num_filters)
                        z_layers.append(z_reshape_noise)


                w = aggregated_feature.get_shape()[1]
                h = aggregated_feature.get_shape()[2]
                c = aggregated_feature.get_shape()[3]
                z1_dense = tf.layers.dense(z_inputs, h * w * c, name='reshape_dense_aggregated')
                z1_reshape = tf.reshape(z1_dense, [self.batch_size, w, h, c])

                outputs = aggregated_feature

                decoder_layers = []
                current_layers = [outputs]

                with tf.variable_scope('g_deconv_layers'):
                    for i in range(len(self.layer_sizes) + 1):
                        if i < 3:  # Pass the injected noise to the first 3 decoder layers for sharper results
                            if is_z2 > 0:
                                outputs = self.bi_attention_module(z_layers[i], outputs, name='noiseLayer_{}'.format(i),
                                                                   training=training)
                            else:
                                outputs = outputs

                        idx = len(self.layer_sizes) - 1 - i
                        num_features = self.layer_sizes[idx]
                        inner_layers = self.inner_layers[idx]
                        upscale_shape = encoder_layers[idx].get_shape().as_list()

                        if idx < 0:
                            num_features = self.layer_sizes[0]
                            inner_layers = self.inner_layers[0]
                            upscale_shape = support_input[:, 0].get_shape().as_list()

                        with tf.variable_scope('g_deconv{}'.format(i)):
                            decoder_inner_layers = [outputs]
                            for j in range(inner_layers):
                                if i == 0 and j == 0:
                                    with tf.variable_scope('g_deconv_innner_deconv{}'.format(j)):
                                        outputs = self.add_decoder_layer(input=outputs,
                                                                         name="decoder_inner_conv_{}_{}"
                                                                         .format(i, j),
                                                                         training=training,
                                                                         layer_to_skip_connect=current_layers,
                                                                         num_features=num_features,
                                                                         dim_upscale=False,
                                                                         local_inner_layers=decoder_inner_layers,
                                                                         dropout_rate=dropout_rate,
                                                                         scope="decoder_inner_conv_{}_{}"
                                                                         .format(i, j), )
                                        decoder_inner_layers.append(outputs)
                                else:
                                    with tf.variable_scope('g_deconv_innner_deconv{}'.format(j)):
                                        outputs = self.add_decoder_layer(input=outputs,
                                                                         name="decoder_inner_conv_{}_{}"
                                                                         .format(i, j), training=training,
                                                                         layer_to_skip_connect=current_layers,
                                                                         num_features=num_features,
                                                                         dim_upscale=False,
                                                                         local_inner_layers=decoder_inner_layers,
                                                                         w_size=upscale_shape[1],
                                                                         h_size=upscale_shape[2],
                                                                         dropout_rate=dropout_rate,
                                                                         scope="decoder_inner_conv_{}_{}"
                                                                         .format(i, j))
                                        decoder_inner_layers.append(outputs)

                            if idx >= 0:
                                upscale_shape = encoder_layers[idx - 1].get_shape().as_list()
                                if idx == 0:
                                    upscale_shape = support_input[:, 0].get_shape().as_list()
                                outputs = self.add_decoder_layer(
                                    input=outputs,
                                    name="decoder_outer_conv_{}".format(i),
                                    training=training,
                                    layer_to_skip_connect=current_layers,
                                    num_features=num_features,
                                    dim_upscale=True, local_inner_layers=decoder_inner_layers, w_size=upscale_shape[1],
                                    h_size=upscale_shape[2], dropout_rate=dropout_rate,
                                    scope="decoder_outer_conv_{}".format(i))
                                current_layers.append(outputs)

                            current_layers.append(outputs)
                            decoder_layers.append(outputs)
                            print('decoder_{}_outputs'.format(idx), outputs)


                            connection_layers = 2
                            if len(self.layer_sizes) == 6:
                                if connection_layers == 3:
                                    connection_condition = (idx - 3)
                                elif connection_layers == 2:
                                    connection_condition = (idx - 4)
                                elif connection_layers == 1:
                                    connection_condition = (idx - 5)
                                elif connection_layers == 0:
                                    connection_condition = -1
                                else:
                                    raise NameError
                            elif len(self.layer_sizes) == 4:
                                if connection_layers == 6:
                                    connection_condition = -(idx + 2)
                                elif connection_layers == 5:
                                    connection_condition = -(idx + 1)
                                elif connection_layers == 4:
                                    connection_condition = -(idx)
                                elif connection_layers == 3:
                                    connection_condition = (idx - 1)
                                elif connection_layers == 2:
                                    connection_condition = (idx - 2)
                                elif connection_layers == 1:
                                    connection_condition = (idx - 3)
                                elif connection_layers == 0:
                                    connection_condition = -1
                                else:
                                        raise NameError
                            elif len(self.layer_sizes) == 5:
                                if connection_layers == 3:
                                    connection_condition = (idx - 2)
                                elif connection_layers == 2:
                                    connection_condition = (idx - 3)
                                elif connection_layers == 1:
                                    connection_condition = idx - 4
                                elif connection_layers == 0:
                                    connection_condition = -1
                                else:
                                    raise NameError
                            else:
                                raise NameError


                            if connection_condition >= 0:
                                attention_slim = []
                                encoder_s_attention = []
                                for s in range(support_num):
                                    current_sim = tf.expand_dims(
                                        tf.expand_dims(tf.expand_dims(similarities[:, s], -1), -1), -1)
                                    current_encoder = tf.multiply(current_sim, encoder_layers_whole[s][idx - 1])
                                    # attention_feature, attention = contextual_attention(outputs, current_encoder,mask=None, ksize=3, stride=1, rate=2)
                                    attention_feature, attention = self.attention(current_encoder, outputs,
                                                                                  current_encoder.get_shape()[-1], True,
                                                                                  scope='attention_{}'.format(s))
                                    # attention_feature, attention = contextual_attention(current_encoder,outputs, mask=None, ksize=3, stride=1, rate=2)
                                    ##### attention map visualization ######
                                    fs = tf.shape(current_encoder)
                                    int_fs = current_encoder.get_shape().as_list()
                                    bs = tf.shape(outputs)
                                    int_bs = outputs.get_shape().as_list()
                                    attention_reshape = tf.reshape(attention, [-1, fs[1], fs[2], bs[1] * bs[2]])
                                    offset = tf.argmax(attention_reshape, axis=3, output_type=tf.int32)
                                    offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
                                    offset.set_shape(int_bs[:3] + [2])
                                    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
                                    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
                                    offset = offset - tf.concat([h_add, w_add], axis=3)
                                    flow = flow_to_image_tf(offset)
                                    attention_sli = flow
                                    attention_sli = tf.image.resize_images(attention_sli,
                                                                           size=[support_input.get_shape()[2],
                                                                                 support_input.get_shape()[2]],
                                                                           method=ResizeMethod.BILINEAR)
                                    fusion = 0.8 * attention_sli + 0.2 * support_input[:, s]

                                    selected_fusion = [fusion[0]]
                                    selected_fusion = tf.concat(selected_fusion, axis=1)
                                    attention_slim.append(selected_fusion)
                                    encoder_s_attention.append(attention_feature)

                                attention_image = tf.concat(attention_slim, axis=0)
                                attention_image = tf.expand_dims(attention_image, axis=0)
                                # tf.summary.image('{}'.format(idx - 1), attention_image)
                                ##### for xsi and xb
                                encoder_s_attention = tf.stack(encoder_s_attention, axis=1)
                                encoder_s_attention = tf.reduce_sum(encoder_s_attention, axis=1)
                                outputs = tf.concat([outputs, encoder_s_attention], axis=3)
                                current_layers[-1] = outputs

                    high_res_layers = []

                    for p in range(2):
                        outputs = self.conv_layer(outputs, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                                  transpose=False, scope='p_conv_{}'.format(p))
                        outputs = leaky_relu(features=outputs)
                        outputs = batch_norm(outputs,
                                             decay=0.99, scale=True,
                                             center=True, is_training=training,
                                             renorm=True, scope='p_bn_{}'.format(p))
                        high_res_layers.append(outputs)

                    outputs = self.conv_layer(outputs, self.num_channels, [3, 3], strides=(1, 1), sn=False,
                                              transpose=False, scope='Last_conv')

                # output images
                with tf.variable_scope('g_tanh'):
                    gan_decoder = tf.tanh(outputs, name='outputs')
                    support_images = []
                    for s in range(support_num):
                        support_images.append(
                            tf.concat([support_input[0, s]], axis=1))
                    support_images = tf.concat(support_images, axis=0)

                    generated_images = tf.concat([gan_decoder[0]], axis=1)
                    real_images = tf.concat([conditional_input[0]], axis=1)

                    compare_image = tf.concat([support_images, generated_images], axis=0)
                    compare_image = tf.expand_dims(compare_image, axis=0)
                    # tf.summary.image('comparison', compare_image)

                with tf.variable_scope('reconstruction_layer'):
                    S_l1 = []
                    for s in range(support_num):
                        current_sim = tf.expand_dims(tf.expand_dims(tf.expand_dims(similarities[:, s], -1), -1), -1)
                        current_support_image = tf.multiply(current_sim, support_input[:, s])

                        S_l1.append(tf.abs(gan_decoder - current_support_image))

                    S_l1 = tf.stack(S_l1, axis=1)
                    S_l1 = tf.reduce_sum(S_l1, axis=1)
                    S_l1 = tf.reduce_mean(S_l1)
                    reconstruction_loss = S_l1

            if self.training:
                self.variables_fzl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='few_shot_classifier')
                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

                if self.build:
                    count_parameters(self.variables_fzl, name="few-shot_classifier_parameter_num")
                    print("generator_total_layers", self.conv_layer_num)
                    count_parameters(self.variables, name="generator_parameter_num")
                self.build = False
                return gan_decoder, z_inputs_2, aggregated_feature, similarities, z_inputs, z_inputs, KL_loss, reconstruction_loss, z_inputs, z_inputs, z_inputs, z_inputs, z_inputs


            else:
                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

                if self.build:
                    print("generator_total_layers", self.conv_layer_num)
                    count_parameters(self.variables, name="generator_parameter_num")
                    print(self.variables)
                self.build = False
                return gan_decoder, similarities, _, _, _, _, _, _


def conditional_batchnorm(x, train_phase, scope_bn, y=None, nums_class=10):
    # Batch Normalization
    # Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        if y == None:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[nums_class, x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[nums_class, x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            beta, gamma = tf.nn.embedding_lookup(beta, y), tf.nn.embedding_lookup(gamma, y)
            beta = tf.reshape(beta, [-1, 1, 1, x.shape[-1]])
            gamma = tf.reshape(gamma, [-1, 1, 1, x.shape[-1]])
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments', keep_dims=True)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def global_sum_pooling(inputs):
    inputs = tf.reduce_sum(inputs, [1, 2], keep_dims=False)
    return inputs


# def Inner_product(global_pooled, y, nums_class, update_collection=None):
#     # print(nums_class)
#     # W = global_pooled.shape[-1]
#     # V = tf.get_variable("V", [nums_class, W], initializer=tf.glorot_uniform_initializer())
#     # V = tf.transpose(V)
#     # V = spectral_normalization("embed", V, update_collect ion=update_collection)
#     # V = tf.transpose(V)
#     # y = tf.cast(y,dtype=tf.int32)
#     # temp = tf.nn.embedding_lookup(V, y)
#     temp = tf.reduce_sum(y * global_pooled, axis=1, keep_dims=True)
#     return temp
def spectral_normalization(name, weights, num_iters=1, update_collection=None,
                           with_sigma=False):
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable(name + 'u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def Inner_product(global_pooled, y, y_weights, nums_class, update_collection=None):
    W = global_pooled.shape[-1]
    V = tf.get_variable("V", [nums_class, W], initializer=tf.glorot_uniform_initializer())
    V = tf.transpose(V)
    V = spectral_normalization("embed", V, update_collection=update_collection)
    V = tf.transpose(V)
    temp = tf.nn.embedding_lookup(V, y)
    temp = tf.reduce_sum(y_weights * temp * global_pooled, axis=1, keep_dims=True)

    return temp


def _l2normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def pytorch_kaiming_weight_factor(a=0.0, activation_function='leaky_relu', uniform=False):
    if activation_function == 'relu':
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function == 'tanh':
        gain = 5.0 / 3
    else:
        gain = 1.0

    if uniform:
        factor = gain * gain
        mode = 'FAN_IN'
    else:
        factor = (gain * gain) / 1.3
        mode = 'FAN_IN'

    return factor, mode, uniform


import tensorflow.contrib as tf_contrib

factor, mode, uniform = pytorch_kaiming_weight_factor(a=0.0, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)

weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
weight_regularizer_fully = tf.contrib.layers.l2_regularizer(0.0001)


def spectral_norm_d(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def conv_d(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm_d(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv_d(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv_d(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = instance_norm(x)

        return x + x_init


def pre_resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        _, _, _, init_channel = x_init.get_shape().as_list()

        with tf.variable_scope('res1'):
            x = lrelu(x_init, 0.2)
            x = conv_d(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            x = lrelu(x, 0.2)
            x = conv_d(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        if init_channel != channels:
            with tf.variable_scope('shortcut'):
                x_init = conv_d(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)

        return x + x_init


def adaptive_resblock(x_init, channels, gamma1, beta1, gamma2, beta2, use_bias=True, sn=False,
                      scope='adaptive_resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv_d(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, gamma1, beta1)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv_d(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, gamma2, beta2)

        return x + x_init


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def down_sample_avg(x, scale_factor=2):
    return tf.layers.average_pooling2d(x, pool_size=3, strides=scale_factor, padding='SAME')


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def param_free_norm(x, epsilon=1e-5):
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x_std = tf.sqrt(x_var + epsilon)

    return (x - x_mean) / x_std


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    x = param_free_norm(content, epsilon)

    return gamma * x + beta


##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))  # [64, h, w, c]

    return loss


def dense(name, inputs, nums_out, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
    return tf.nn.bias_add(tf.matmul(inputs, W), b)


def google_attention(self, x, channels, scope='attention'):
    with tf.variable_scope(scope):
        batch_size, height, width, num_channels = x.get_shape().as_list()
        f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
        x = gamma * o + x

    return x


class Discriminator:
    def __init__(self, batch_size, layer_sizes, inner_layers, use_wide_connections=False, name="d"):
        """
        Initialize a discriminator network.
        :param batch_size: Batch size for discriminator.
        :param layer_sizes: A list with the feature maps for each MultiLayer.
        :param inner_layers: An integer indicating the number of inner layers.
        """
        self.training = True
        self.print = False
        self.reuse = tf.AUTO_REUSE
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.use_wide_connections = use_wide_connections
        self.build = True
        self.name = name
        self.classify = AttentionalClassify()
        self.sn = True

    def upscale(self, x, scale):
        """
            Upscales an image using nearest neighbour
            :param x: Input image
            :param h_size: Image height size
            :param w_size: Image width size
            :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h * scale, w * scale))

    def feature_extractor(self, inputs, channel, scope):
        with tf.variable_scope(scope, reuse=self.reuse):
            matching_features = []
            x = conv(inputs, channel, kernel=7, stride=1, pad=3, pad_type='reflect', sn=self.sn, scope='conv')
            matching_features.append(x)
            for i in range(4):
                x = pre_resblock(x, channel * 2, sn=self.sn, scope='front_resblock_0_' + str(i))
                x = pre_resblock(x, channel * 2, sn=self.sn, scope='front_resblock_1_' + str(i))
                x = down_sample_avg(x, scale_factor=2)
                channel = channel * 2
                matching_features.append(x)

            # channel = 1024
            for i in range(2):
                x = pre_resblock(x, channel, sn=self.sn, scope='back_resblock_' + str(i))
                matching_features.append(x)
            return x, matching_features

    def classify(self, similarities, support_set_y, name, training=False):
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification',
                                                                                   reuse=self.reuse):
            preds = tf.squeeze(tf.matmul(tf.expand_dims(similarities, 1), support_set_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds

    def __call__(self, conditional_images, target_image1, target_image2, z1_1, z1_2, sim_label, similarities_index, y_global_batch,
                 y_global_support, selected_classes,
                 support_num,
                 classes, similarities1, similarities2, z_1, training=False, dropout_rate=0.0):
        """
        :param conditional_input: A batch of conditional inputs (x_i) of size [batch_size, height, width, channel]
        :param generated_input: A batch of generated inputs (x_g) of size [batch_size, height, width, channel]
        :param training: Placeholder for training or a boolean indicating training or validation
        :param dropout_rate: A float placeholder for dropout rate or a float indicating the dropout rate
        :param name: Network name
        :return:
        """
        # generated_input = tf.convert_to_tensor(generated_input)
        with tf.variable_scope(self.name, reuse=self.reuse):
            channel = 32
            real_features = []
            real_features_avg = []
            real_matching_features = []

            ##### obtaining real features
            for k, image in enumerate(tf.unstack(conditional_images, axis=1)):
                feature, real_matching_feature = self.feature_extractor(image, channel, 'feature_extractor')
                real_matching_features.append(real_matching_feature)
                real_features.append(feature)
                real_features_avg.append(tf.reduce_mean(feature, axis=[1, 2]))


            #### obtaining fake features
            fake_features = []
            fake_features_avg = []
            fake_matching_features = []
            target_images = [target_image1, target_image2]
            for i in range(2):
                feature, fake_matching_feature = self.feature_extractor(target_images[i], channel, 'feature_extractor')
                fake_matching_features.append(fake_matching_feature)
                fake_features.append(feature)
                fake_features_avg.append(tf.reduce_mean(feature, axis=[1, 2]))
           


            ##### classification
            classification_fakes = []
            for i in range(2):
                fake_logit = tf.layers.dense(fake_features_avg[i], units=classes, name='classifier')
                classification_fake = tf.nn.softmax_cross_entropy_with_logits(labels=y_global_batch, logits=fake_logit)
                classification_fakes.append(classification_fake)
            classification_fakes = tf.reduce_mean(tf.stack(classification_fakes, axis=1),axis=1)

            classification_reals = []
            for i in range(support_num):
                real_logit = tf.layers.dense(real_features_avg[i], units=classes, name='classifier')
                classification_real = tf.nn.softmax_cross_entropy_with_logits(labels=y_global_batch, logits=real_logit)
                classification_reals.append(classification_real)
            classification_reals = tf.reduce_mean(tf.stack(classification_reals, axis=1),axis=1)
           ##### classification


            ##### adversarial loss
            d_fakes = []
            for i in range(2):
                d_fake = tf.layers.dense(fake_features_avg[i], units=1, name='dense_fakereal')
                d_fakes.append(d_fake)
            d_fakes = tf.reduce_mean(tf.stack(d_fakes, axis=1),axis=1)

            d_reals = []
            for i in range(support_num):
                d_real = tf.layers.dense(real_features_avg[i], units=1, name='dense_fakereal')
                d_reals.append(d_real)
            d_reals = tf.reduce_mean(tf.stack(d_reals, axis=1),axis=1)
            ##### adversarial loss


            ##### weighted feature loss
            feature_total = tf.stack(real_features, axis=1)
            feature_total_reshape = tf.reshape(feature_total,
                                               [feature_total.get_shape()[0], feature_total.get_shape()[1],
                                                feature_total.get_shape()[2] * \
                                                feature_total.get_shape()[3] * feature_total.get_shape()[4]])
            feature_total_reshape1 = self.classify(similarities1, support_set_y=feature_total_reshape, name='classify')
            feature_total_reshape2 = self.classify(similarities2, support_set_y=feature_total_reshape, name='classify')
            feature_total1 = tf.reshape(feature_total_reshape1,
                                        [feature_total.get_shape()[0], feature_total.get_shape()[2],
                                         feature_total.get_shape()[3], feature_total.get_shape()[4]])
            feature_total2 = tf.reshape(feature_total_reshape2,
                                        [feature_total.get_shape()[0], feature_total.get_shape()[2],
                                         feature_total.get_shape()[3], feature_total.get_shape()[4]])
            feature_loss1 = tf.abs(feature_total1 - fake_features[0])
            feature_loss2 = tf.abs(feature_total2 - fake_features[1])
            feature_loss = tf.reduce_mean(tf.stack([feature_loss1, feature_loss2], axis=1),axis=1)
            ##### weighted feature loss


            ##### mode loss over different resolution
            # mode_losses= []
            # for i in range(len(fake_matching_features[0])):
            #     mode_loss = tf.divide(tf.reduce_mean(tf.abs(similarities1 - similarities2), axis=[1]) + 1e-6,
            #                                                     tf.reduce_mean(tf.abs(fake_matching_features[0][i] - fake_matching_features[1][i]), axis=[1, 2, 3]))
            #     mode_losses.append(mode_loss)
            # mode_losses = tf.reduce_mean(tf.stack(mode_losses, axis=1),axis=1)

            mode_loss = tf.reduce_mean(tf.abs(fake_features[0] - fake_features[1]), axis=[1, 2, 3]) / tf.reduce_mean(
                tf.abs(similarities1 - similarities2), axis=[1])



            ##### regressing the interpolation coefficients
            sim_losses = []
            similarities = [similarities1, similarities2]
            for i in range(2):
                real_features_combine = tf.concat(real_features,axis=3)
                combined_feature = tf.concat([real_features_combine, fake_features[i]], axis=3)
                combine_channel = 256
                for j in range(3):
                    combined_feature = tf.layers.dense(combined_feature, units=combine_channel, name='dense_{}'.format(j))
                    combine_channel = combine_channel / 2
                combined_feature = tf.reduce_mean(combined_feature, axis=[1, 2])
                combine_x = tf.layers.dense(combined_feature, units=support_num, name='predict_sim')
                predict_sim = tf.nn.softmax(combine_x)
                sim_losses.append(tf.abs(predict_sim - similarities[i]))
            sim_losses = tf.reduce_mean(tf.stack(sim_losses, axis=1),axis=1)



        self.reuse = True
        self.variables_shared = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.variables_d = self.variables_shared

        if self.build:
            count_parameters(self.variables_shared, name="discriminator_parameters_num")
        self.build = False

        return d_reals, d_fakes, feature_loss, classification_reals, classification_fakes, sim_losses, mode_loss

