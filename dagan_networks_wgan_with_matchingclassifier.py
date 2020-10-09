import tensorflow as tf
from dagan_architectures_with_matchingclassifier import UResNetGenerator, Discriminator, gaussian_noise_layer, \
    gaussian_noise_change_layer
import numpy as np
import time


def Hinge_loss(real_logits, fake_logits):
    D_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_logits)) - tf.reduce_mean(tf.minimum(0., -1.0 - fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    return D_loss, G_loss


class DAGAN:
    def __init__(self, input_x_i, input_x_j, input_y_i, input_y_j, input_global_y_i, input_global_y_j,
                 input_x_j_selected, input_global_y_j_selected, classes, dropout_rate, generator_layer_sizes,
                 discriminator_layer_sizes, generator_layer_padding, z_inputs, z_inputs_2, matching, fce,
                 full_context_unroll_k, average_per_class_embeddings, batch_size=100, z_dim=100,
                 num_channels=1, is_training=True, augment=True, discr_inner_conv=0, gen_inner_conv=0, num_gpus=1,
                 is_z2=True, is_z2_vae=True,
                 use_wide_connections=False, selected_classes=5, support_num=5, loss_G=1, loss_D=1, loss_KL=0.0001,
                 loss_recons_B=0.01, loss_matching_G=0.01, loss_matching_D=0.01, loss_CLA=1, loss_FSL=1, loss_sim=0.01,
                 z1z2_training=True):

        """
        Initializes a DAGAN object.
        :param input_x_i: Input image x_i
        :param input_x_j: Input image x_j
        :param dropout_rate: A dropout rate placeholder or a scalar to use throughout the network
        :param generator_layer_sizes: A list with the number of feature maps per layer (generator) e.g. [64, 64, 64, 64]
        :param discriminator_layer_sizes: A list with the number of feature maps per layer (discriminator)
                                                                                                   e.g. [64, 64, 64, 64]
        :param generator_layer_padding: A list with the type of padding per layer (e.g. ["SAME", "SAME", "SAME","SAME"]
        :param z_inputs: A placeholder for the random noise injection vector z (usually gaussian or uniform distribut.)
        :param batch_size: An integer indicating the batch size for the experiment.
        :param z_dim: An integer indicating the dimensionality of the random noise vector (usually 100-dim).
        :param num_channels: Number of image channels
        :param is_training: A boolean placeholder for the training/not training flag
        :param augment: A boolean placeholder that determines whether to augment the data using rotations
        :param discr_inner_conv: Number of inner layers per multi layer in the discriminator
        :param gen_inner_conv: Number of inner layers per multi layer in the generator
        :param num_gpus: Number of GPUs to use for training
        """
        self.training = True
        self.print = False
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.z_inputs = z_inputs
        self.z_inputs_2 = z_inputs_2
        self.num_gpus = num_gpus
        self.support_num = support_num
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.loss_KL = loss_KL
        self.loss_CLA = loss_CLA
        self.loss_FSL = loss_FSL
        self.loss_matching_G = loss_matching_G
        self.loss_recons_B = loss_recons_B
        self.loss_matching_D = loss_matching_D
        self.loss_sim = loss_sim
        self.input_x_i = input_x_i
        self.input_x_j = input_x_j
        self.input_x_j_selected = input_x_j_selected
        self.input_y_i = input_y_i
        self.input_y_j = input_y_j
        self.input_global_y_i = input_global_y_i
        self.input_global_y_j = input_global_y_j
        self.input_global_y_j_selected = input_global_y_j_selected
        self.classes = classes
        self.selected_classes = selected_classes
        self.dropout_rate = dropout_rate
        self.training_phase = is_training
        self.augment = augment
        self.is_z2 = is_z2
        self.is_z2_vae = is_z2_vae
        self.z1z2_training = z1z2_training

        self.g = UResNetGenerator(batch_size=self.batch_size, layer_sizes=generator_layer_sizes,
                                  num_channels=num_channels, layer_padding=generator_layer_padding,
                                  inner_layers=gen_inner_conv, name="generator", matching=matching, fce=fce,
                                  full_context_unroll_k=full_context_unroll_k,
                                  average_per_class_embeddings=average_per_class_embeddings)

        self.d = Discriminator(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
                               inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections,
                               name="discriminator")

    def rotate_data(self, image_a, image_b):
        """
        Rotate 2 images by the same number of degrees
        :param image_a: An image a to rotate k degrees
        :param image_b: An image b to rotate k degrees
        :return: Two images rotated by the same amount of degrees
        """
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image_a = tf.image.rot90(image_a, k=random_variable[0])
        image_b = tf.image.rot90(image_b, k=random_variable[0])
        return [image_a, image_b]

    def rotate_batch(self, batch_images_a, batch_images_b):
        """
        Rotate two batches such that every element from set a with the same index as an element from set b are rotated
        by an equal amount of degrees
        :param batch_images_a: A batch of images to be rotated
        :param batch_images_b: A batch of images to be rotated
        :return: A batch of images that are rotated by an element-wise equal amount of k degrees
        """
        shapes = map(int, list(batch_images_a.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked_a = tf.unstack(batch_images_a)
            batch_images_unpacked_b = tf.unstack(batch_images_b)
            new_images_a = []
            new_images_b = []
            for image_a, image_b in zip(batch_images_unpacked_a, batch_images_unpacked_b):
                rotate_a, rotate_b = self.augment_rotate(image_a, image_b)
                new_images_a.append(rotate_a)
                new_images_b.append(rotate_b)

            new_images_a = tf.stack(new_images_a)
            new_images_a = tf.reshape(new_images_a, (batch_size, x, y, c))
            new_images_b = tf.stack(new_images_b)
            new_images_b = tf.reshape(new_images_b, (batch_size, x, y, c))
            return [new_images_a, new_images_b]

    def generate(self, conditional_images, support_input, input_global_x_j_selected, input_y_i, input_y_j,
                 input_global_y_i, input_global_y_j_selected, selected_classes, support_num, classes, is_z2, is_z2_vae,
                 z_input=None, z_input_2=None):
        """
        Generate samples with the DAGAN
        :param conditional_images: Images to condition DAGAN on.
        :param z_input: Random noise to condition the DAGAN on. If none is used then the method will generate random
        noise with dimensionality [batch_size, z_dim]
        :return: A batch of generated images, one per conditional image
        """
        if z_input is None:
            z_input = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)
            z_input_2 = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)
        if self.training:
            generated_samples, z1, aggregated_feature, similarities, similarities_data, loss_recg, KL_loss, reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = self.g(
                z_input, z_input_2,
                conditional_images, support_input, input_y_i, input_y_j, selected_classes, support_num, is_z2,
                is_z2_vae,
                training=self.training_phase,
                dropout_rate=self.dropout_rate,
                z1z2_training=self.z1z2_training,
                z_dim=self.z_dim)

            return generated_samples, z1, aggregated_feature, similarities, similarities_data, z_input, z_input_2, loss_recg, KL_loss, reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake
        else:
            generated_samples, similarities, similarities_data, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = self.g(
                z_input, z_input_2,
                conditional_images, support_input, input_y_i, input_y_j, selected_classes, support_num, is_z2,
                is_z2_vae,
                training=self.training_phase,
                dropout_rate=self.dropout_rate,
                z1z2_training=self.z1z2_training,
                z_dim=self.z_dim)

            similarities_onehot = tf.cast((0) * tf.ones_like(similarities[:, 0]), dtype=tf.int32)
            similarities_onehot = tf.expand_dims(similarities_onehot, axis=-1)
            similarities_index = tf.expand_dims(similarities[:, 0], axis=-1)

            g_same_class_outputs = self.d(generated_samples, similarities_onehot, similarities_index,
                                          input_global_x_j_selected, input_global_y_i,
                                          input_global_y_j_selected, selected_classes, support_num, classes,
                                          similarities, training=self.training_phase,
                                          dropout_rate=self.dropout_rate)
            return generated_samples, similarities, g_same_class_outputs, preds_fake

    def augment_rotate(self, image_a, image_b):
        r = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None))
        rotate_boolean = tf.equal(0, r, name="check-rotate-boolean")
        [image_a, image_b] = tf.cond(rotate_boolean[0], lambda: self.rotate_data(image_a, image_b),
                                     lambda: [image_a, image_b])
        return image_a, image_b

    def data_augment_batch(self, batch_images_a, batch_images_b):
        """
        Apply data augmentation to a set of image batches if self.augment is set to true
        :param batch_images_a: A batch of images to augment
        :param batch_images_b: A batch of images to augment
        :return: A list of two augmented image batches
        """

        [images_a, images_b] = tf.cond(self.augment, lambda: self.rotate_batch(batch_images_a, batch_images_b),
                                       lambda: [batch_images_a, batch_images_b])

        return images_a, images_b

    def save_features(self, name, features):
        """
        Save feature activations from a network
        :param name: A name for the summary of the features
        :param features: The features to save
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = 8
            x_channels = channels / y_channels

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                  y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            # tf.summary.image('{}_{}'.format(name, i), activations_features)

    def loss(self, gpu_id):

        """
        Builds models, calculates losses, saves tensorboard information.
        :param gpu_id: The GPU ID to calculate losses for.
        :return: Returns the generator and discriminator losses.
        """
        #### general matching procedure
        with tf.name_scope("losses_{}".format(gpu_id)):
            before_loss = time.time()
            epsilon = 1e-8
            input_a, input_b, input_y_a, input_y_b, input_global_y_a, input_global_y_b, input_b_selected, input_global_y_b_selected = \
                self.input_x_i[gpu_id], self.input_x_j[gpu_id], self.input_y_i[gpu_id], self.input_y_j[gpu_id], \
                self.input_global_y_i[gpu_id], self.input_global_y_j[gpu_id], self.input_x_j_selected[gpu_id], \
                self.input_global_y_j_selected[gpu_id]

            # input_a_expand = tf.expand_dims(input_a,1)
            # input_a_copy = tf.tile(input_a_expand,[1,self.support_num,1,1,1])
            # current_support = tf.cond(self.z1z2_training,lambda:input_a_copy,lambda:input_b)
            current_support = input_b

            x_g1, z1_1, aggregated_feature1, similarities1, similarities_data, z_input, z_input_2, recg_loss, KL_loss, \
            reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = \
                self.generate(input_a, current_support, input_b_selected, input_y_a, input_y_b, input_global_y_a,
                              input_global_y_b_selected, self.selected_classes, self.support_num, self.classes,
                              self.is_z2, self.is_z2_vae)

            x_g2, z1_2, aggregated_feature2, similarities2, similarities_data, z_input, z_input_2, recg_loss, KL_loss, \
            reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = \
                self.generate(input_a, current_support, input_b_selected, input_y_a, input_y_b, input_global_y_a,
                              input_global_y_b_selected, self.selected_classes, self.support_num, self.classes,
                              self.is_z2, self.is_z2_vae)

            #### diversification loss
            loss_diversification = tf.reduce_mean(tf.abs(x_g1 - x_g2))

            #### mode loss

            #### cycle reconstruction

            feature_total = []
            # similarities_onehot = tf.cast((0) * tf.ones_like(similarities[:, 0]), dtype=tf.int32)
            # similarities_onehot = tf.one_hot(similarities_onehot,self.support_num)
            similarities_onehot = tf.expand_dims(similarities1[:, 0], axis=-1)
            similarities_index = tf.expand_dims(similarities1[:, 0], axis=-1)

            #### fake image
            d_real, d_fake, feature_loss, t_classification_loss, g_classification_loss, sim_loss, mode_loss = self.d(
                input_b, x_g1, x_g2, z1_1, z1_2, similarities_onehot, similarities_index,
                input_global_y_b[:, 0], input_global_y_b_selected,
                self.selected_classes, self.support_num,
                self.classes, similarities1, similarities2, z1_1,
                training=self.training_phase,
                dropout_rate=self.dropout_rate)

            # mode_loss =tf.reduce_mean(mode_feature,axis=[1,2,3]) / tf.reduce_mean(tf.abs(aggregated_feature1 - aggregated_feature2),axis=[1,2,3])

            #### distinguish interpolation coefficients
            d_loss_pure, G_loss = Hinge_loss(d_real, d_fake)


            ##### without mask
            mode_loss = tf.reduce_mean(mode_loss)
            sim_loss = tf.reduce_mean(sim_loss)
            loss_KL = tf.reduce_mean(KL_loss)
            loss_recg = tf.reduce_mean(recg_loss)
            loss_reconstruction = tf.reduce_mean(reconstruction_loss)
            loss_feature = tf.reduce_mean(feature_loss)
            g_classification_loss = tf.reduce_mean(g_classification_loss)
            t_classification_loss = tf.reduce_mean(t_classification_loss)

            g_loss = G_loss * self.loss_G + g_classification_loss * self.loss_CLA + self.loss_recons_B * loss_reconstruction + \
                     self.loss_matching_D * loss_feature + self.loss_sim * sim_loss + self.loss_matching_G * mode_loss

            d_loss = self.loss_D * (d_loss_pure) + self.loss_CLA * t_classification_loss

            # tf.add_to_collection('fzl_losses',crossentropy_loss_real)

            tf.add_to_collection('g_losses', g_loss)
            tf.add_to_collection('d_losses', d_loss)
            tf.add_to_collection('c_losses', t_classification_loss)
            # tf.add_to_collection('recons_loss', recons_loss)

            # tf.summary.scalar('G_pure_losses', G_loss_image)
            # tf.summary.scalar('D_pure_losses', d_loss_image)
            # tf.summary.scalar('G_pairs_losses', G_loss_sim)
            # tf.summary.scalar('D_pairs_losses', d_loss_sim)
            tf.summary.scalar('mode_losses', mode_loss)
            tf.summary.scalar('G_losses', G_loss)
            tf.summary.scalar('D_losses', d_loss_pure)
            tf.summary.scalar('sim_losses', sim_loss)
            # tf.summary.scalar('verification_losses', verification_loss)
            tf.summary.scalar('total_g_losses', g_loss)
            tf.summary.scalar('total_d_losses', d_loss)
            tf.summary.scalar('c_losses', g_classification_loss)
            tf.summary.scalar('reconstruction_losses', loss_reconstruction)
            tf.summary.scalar('matchingD_losses', loss_feature)

        return {
            "g_losses": tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            "d_losses": tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
            "c_losses": tf.add_n(tf.get_collection('c_losses'), name='total_c_loss'),
            # "fzl_losses":tf.add_n(tf.get_collection('fzl_losses'),name='total_fzl_loss'),
            # "recons_losses":tf.add_n(tf.get_collection('recons_losses'),name='total_recons_loss'),
        }

    def train(self, opts, losses):

        """
        Returns ops for training our DAGAN system.
        :param opts: A dict with optimizers.
        :param losses: A dict with losses.
        :return: A dict with training ops for the dicriminator and the generator.
        """
        opt_ops = dict()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"],
                                                         var_list=self.g.variables,
                                                         colocate_gradients_with_ops=True)
            opt_ops["d_opt_op"] = opts["d_opt"].minimize(losses["d_losses"],
                                                         var_list=self.d.variables_d,
                                                         colocate_gradients_with_ops=True)
            opt_ops["c_opt_op"] = opts["c_opt"].minimize(losses['c_losses'], var_list=self.d.variables_d,
                                                         colocate_gradients_with_ops=True)

            # opt_ops["fzl_opt_op"] = opts["fzl_opt"].minimize(losses['fzl_losses'], var_list=self.g.variables_fzl,
            #                                              colocate_gradients_with_ops=True)
            # opt_ops["recons_opt_op"] = opts["recons_opt"].minimize(losses['recons_losses'], var_list=self.g.variables,
            #                                              colocate_gradients_with_ops=True)

        return opt_ops

    def init_train(self, learning_rate=1e-4, beta1=0.0, beta2=0.9):
        """
        Initialize training by constructing the summary, loss and ops
        :param learning_rate: The learning rate for the Adam optimizer
        :param beta1: Beta1 for the Adam optimizer
        :param beta2: Beta2 for the Adam optimizer
        :return: summary op, losses and training ops.
        """

        losses = dict()
        opts = dict()

        if self.num_gpus > 0:
            device_ids = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]
        else:
            device_ids = ['/cpu:0']
        for gpu_id, device_id in enumerate(device_ids):
            with tf.device(device_id):
                total_losses = self.loss(gpu_id=gpu_id)
                for key, value in total_losses.items():
                    if key not in losses.keys():
                        losses[key] = [value]
                    else:
                        losses[key].append(value)

        for key in list(losses.keys()):
            losses[key] = tf.reduce_mean(losses[key], axis=0)
            opts[key.replace("losses", "opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                        learning_rate=learning_rate)

            # opts[key.replace("losses", "opt")] = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        summary = tf.summary.merge_all()
        apply_grads_ops = self.train(opts=opts, losses=losses)

        return summary, losses, apply_grads_ops

    def sample_same_images(self):
        """
        Samples images from the DAGAN using input_x_i as image




        conditional input and z_inputs as the gaussian noise.
        :return: Inputs and generated images
        """
        conditional_inputs = self.input_x_i[0]
        support_input = self.input_x_j[0]
        input_global_y_i = self.input_global_y_i[0]
        input_global_x_j_selected = self.input_x_j_selected[0]

        input_y_i = self.input_y_i[0]
        input_y_j = self.input_y_j[0]
        input_global_y_j = self.input_global_y_j[0]
        input_global_y_j_selected = self.input_global_y_j_selected[0]

        classes = self.classes

        # new_summary = tf.summary.merge_all()

        #### calculating the d_loss for score of selected samples
        # x = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # print('hhhhhhhhhhh',x)

        # new_summary = tf.summary.merge(
        #     [tf.get_collection(tf.GraphKeys.SUMMARIES, '4'), tf.get_collection(tf.GraphKeys.SUMMARIES, '3'),
        #      tf.get_collection(tf.GraphKeys.SUMMARIES, '2')])

        if self.training:
            generated, f_encode_z, matching_feature, similarities, similarities_data, z_input, z_input_2, loss_recg, KL_loss, reconstruction_loss, crossentropy_loss_real, crossentropy_loss_fake, accuracy_real, accuracy_fake, preds_fake = \
                self.generate(conditional_images=conditional_inputs,
                              support_input=support_input,
                              input_global_y_i=input_global_y_i,
                              input_global_x_j_selected=input_global_x_j_selected,
                              input_y_i=input_y_i,
                              input_y_j=input_y_j,
                              input_global_y_j_selected=input_global_y_j_selected,
                              selected_classes=self.selected_classes,
                              support_num=self.support_num,
                              classes=classes,
                              z_input=self.z_inputs,
                              z_input_2=self.z_inputs_2,
                              is_z2=self.is_z2,
                              is_z2_vae=self.is_z2_vae)
            return self.input_x_i[0], self.input_x_j[
                0], generated, generated, generated,  input_y_i, input_global_y_i
        else:
            generated, similarities, d_loss, preds_fake = self.generate(
                conditional_images=conditional_inputs,
                support_input=support_input,
                input_global_y_i=input_global_y_i,
                input_global_x_j_selected=input_global_x_j_selected,
                input_y_i=input_y_i,
                input_y_j=input_y_j,
                input_global_y_j_selected=input_global_y_j_selected,
                selected_classes=self.selected_classes,
                support_num=self.support_num,
                classes=classes,
                z_input=self.z_inputs,
                z_input_2=self.z_inputs_2,
                is_z2=self.is_z2,
                is_z2_vae=self.is_z2_vae)

            # print('here',preds_fake) (16, 5)
            # few_shot_fake_category = tf.argmax(preds_fake, axis=1)
            # softmax = tf.nn.softmax(preds_fake)
            # few_shot_confidence_score = tf.reduce_max(softmax, axis=1)

            # print('11111',few_shot_fake_category) shape=(16,)
            # print('22222',few_shot_confidence_score) shape=(16,)

            return self.input_x_i[0], self.input_x_j[
                0], generated, input_y_i, input_global_y_i, similarities, similarities, similarities

    def summary(self):
        new_summary = tf.summary.merge_all()
        return new_summary

    def sampler(self):
        new_summary = tf.summary.merge_all()
        return new_summary, self.sample_same_images()



