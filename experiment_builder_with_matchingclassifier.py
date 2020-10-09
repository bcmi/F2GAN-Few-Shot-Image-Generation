import utils.interpolations as interpolations
import numpy as np
import tqdm
from utils.storage import save_statistics, build_experiment_folder
from tensorflow.contrib import slim

from dagan_networks_wgan_with_matchingclassifier import *
from utils.sampling_with_matchingclassifier import sample_generator, sample_two_dimensions_generator
import time


def isNaN(num):
    return num != num


class ExperimentBuilder(object):
    def __init__(self, args, data):
        tf.reset_default_graph()
        self.continue_from_epoch = args.continue_from_epoch
        self.experiment_name = args.experiment_title
        self.saved_models_filepath, self.log_path, self.save_image_path = build_experiment_folder(self.experiment_name)
        self.num_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        # self.support_number = args.support_number
        self.selected_classes = args.selected_classes
        gen_depth_per_layer = args.generator_inner_layers
        discr_depth_per_layer = args.discriminator_inner_layers
        self.z_dim = args.z_dim
        self.num_generations = args.num_generations
        self.dropout_rate_value = args.dropout_rate_value
        self.data = data
        self.reverse_channels = False

        if args.generation_layers == 6:
            ### animals
            generator_layers = [64, 64, 128, 128, 256, 256]
            #flowers
            #generator_layers = [64, 64, 96, 96, 128, 128]
            gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer,
                                gen_depth_per_layer, gen_depth_per_layer]
            generator_layer_padding = ["SAME", "SAME", "SAME", "SAME", "SAME", "SAME"]
        else:
            generator_layers = [64, 64, 128, 128]
            gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer,
                                gen_depth_per_layer, gen_depth_per_layer]
            generator_layer_padding = ["SAME", "SAME", "SAME", "SAME"]

        


        #### class encoder channel dimension 64
     
        discriminator_layers = [64, 64, 128, 128]
        discr_inner_layers = [discr_depth_per_layer, discr_depth_per_layer, discr_depth_per_layer,
                              discr_depth_per_layer]

        image_height = data.image_height
        image_width = data.image_width
        image_channel = data.image_channel
        self.image_width = image_width
        self.classes_encode_dimenstion = generator_layers[0]

        self.classes = tf.placeholder(tf.int32)
        self.selected_classes = tf.placeholder(tf.int32)
        self.support_number = tf.placeholder(tf.int32)

        #### [self.input_x_i, self.input_y_i, self.input_global_y_i] --> [images, few shot label, global label]
        ## batch: [self.input_x_i, self.input_y_i, self.input_global_y_i]
        ## support: self.input_x_j, self.input_y_j, self.input_global_y_j]
        ## the input of discriminator: [self.input_x_j_selected, self.input_global_y_j_selected]
        self.input_x_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channel], 'batch')
        self.input_y_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.selected_classes],
                                        'y_inputs_bacth')
        self.input_global_y_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, self.data.training_classes],
                                               'y_inputs_bacth_global')

        self.input_x_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size,
                                                     self.data.selected_classes * self.data.support_number,
                                                     image_height, image_width,
                                                     image_channel], 'support')
        self.input_y_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size,
                                                     self.data.selected_classes * self.data.support_number,
                                                     self.data.selected_classes], 'y_inputs_support')
        self.input_global_y_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size,
                                                            self.data.selected_classes * self.data.support_number,
                                                            self.data.training_classes], 'y_inputs_support_global')

        self.input_x_j_selected = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                              image_channel], 'support_discriminator')
        self.input_global_y_j_selected = tf.placeholder(tf.float32,
                                                        [self.num_gpus, self.batch_size, self.data.training_classes],
                                                        'y_inputs_support_discriminator')

        # self.z_input = tf.placeholder(tf.float32, [self.batch_size*self.data.selected_classes, self.z_dim], 'z-input')
        # self.z_input_2 = tf.placeholder(tf.float32, [self.batch_size*self.data.selected_classes, self.z_dim], 'z-input_2')

        self.z_input = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input')
        self.z_input_2 = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input_2')

        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.z1z2_training = tf.placeholder(tf.bool, name='z1z2_training-flag')
        self.random_rotate = tf.placeholder(tf.bool, name='rotation-flag')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')
        self.is_z2 = args.is_z2
        self.is_z2_vae = args.is_z2_vae

        self.matching = args.matching
        self.fce = args.fce
        self.full_context_unroll_k = args.full_context_unroll_k
        self.average_per_class_embeddings = args.average_per_class_embeddings
        self.restore_path = args.restore_path

        self.is_z2 = args.is_z2
        self.is_z2_vae = args.is_z2_vae
        self.loss_G = args.loss_G
        self.loss_D = args.loss_D
        self.loss_CLA = args.loss_CLA
        self.loss_FSL = args.loss_FSL
        self.loss_KL = args.loss_KL
        self.loss_recons_B = args.loss_recons_B
        self.loss_matching_G = args.loss_matching_G
        self.loss_matching_D = args.loss_matching_D
        self.loss_sim = args.loss_sim
        self.strategy = args.strategy

        #### training/validation/testin

        time_1 = time.time()
        dagan = DAGAN(batch_size=self.batch_size, input_x_i=self.input_x_i, input_x_j=self.input_x_j,
                      input_y_i=self.input_y_i, input_y_j=self.input_y_j, input_global_y_i=self.input_global_y_i,
                      input_global_y_j=self.input_global_y_j,
                      input_x_j_selected=self.input_x_j_selected,
                      input_global_y_j_selected=self.input_global_y_j_selected, \
                      selected_classes=self.data.selected_classes, support_num=self.data.support_number,
                      classes=self.data.training_classes,
                      dropout_rate=self.dropout_rate, generator_layer_sizes=generator_layers,
                      generator_layer_padding=generator_layer_padding, num_channels=data.image_channel,
                      is_training=self.training_phase, augment=self.random_rotate,
                      discriminator_layer_sizes=discriminator_layers,
                      discr_inner_conv=discr_inner_layers, is_z2=self.is_z2, is_z2_vae=self.is_z2_vae,
                      gen_inner_conv=gen_inner_layers, num_gpus=self.num_gpus, z_dim=self.z_dim, z_inputs=self.z_input,
                      z_inputs_2=self.z_input_2,
                      use_wide_connections=args.use_wide_connections, fce=self.fce, matching=self.matching,
                      full_context_unroll_k=self.full_context_unroll_k,
                      average_per_class_embeddings=self.average_per_class_embeddings,
                      loss_G=self.loss_G, loss_D=self.loss_D, loss_KL=self.loss_KL, loss_recons_B=self.loss_recons_B,
                      loss_matching_G=self.loss_matching_G, loss_matching_D=self.loss_matching_D,
                      loss_CLA=self.loss_CLA, loss_FSL=self.loss_FSL, loss_sim=self.loss_sim,
                      z1z2_training=self.z1z2_training)
        time_2 = time.time()
        # print('time for constructing graph:',time_2 - time_1)

        self.summary, self.losses, self.graph_ops = dagan.init_train()
        # print('time for initializing graph:',time.time() - time_2)

        # generated image with z and conditional information
        self.same_images = dagan.sample_same_images()

        self.total_train_batches = int(data.training_data_size / (self.batch_size * self.num_gpus))

        self.total_gen_batches = int(data.testing_data_size / (self.batch_size * self.num_gpus))

        self.init = tf.global_variables_initializer()

        self.tensorboard_update_interval = int(self.total_train_batches / 1 / self.num_gpus)
        self.total_epochs = 600

        # if self.continue_from_epoch == -1:
        #     save_statistics(self.log_path, ['epoch', 'total_d_train_loss_mean', 'total_d_val_loss_mean',
        #                                     'total_d_train_loss_std', 'total_d_val_loss_std',
        #                                     'total_g_train_loss_mean', 'total_g_val_loss_mean',
        #                                     'total_g_train_loss_std', 'total_g_val_loss_std'], create=True)

    def run_experiment(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # with tf.Session() as sess:
            time_4 = time.time()
            sess.run(self.init)
            print('time for initializing global parameters:', time.time() - time_4)

            # self.train_writer = tf.summary.FileWriter("{}/train_logs/".format(self.log_path),
            #                                           graph=tf.get_default_graph())
            # self.validation_writer = tf.summary.FileWriter("{}/validation_logs/".format(self.log_path),
            #                                                graph=tf.get_default_graph())

            log_name = "z2vae{}_z2{}_g{}_d{}_kl{}_cla{}_fzl{}_reconsB{}_matchingG{}_matchingD{}_sim{}_Net_batchsize{}_classencodedim{}_imgsize{}".format(
                self.is_z2_vae, self.is_z2, self.loss_G, self.loss_D, self.loss_KL, self.loss_CLA, self.loss_FSL,
                self.loss_recons_B, self.loss_matching_G, self.loss_matching_D, self.loss_sim, self.batch_size,
                self.z_dim, self.image_width)

            self.train_writer = tf.summary.FileWriter("{}/train_logs/{}".format(self.log_path, log_name),
                                                      graph=sess.graph)
            self.validation_writer = tf.summary.FileWriter("{}/validation_logs/{}".format(self.log_path, log_name),
                                                           graph=sess.graph)

            self.train_saver = tf.train.Saver()
            self.val_saver = tf.train.Saver()

            # variable_names = [v.name for v in tf.trainable_variables()]
            # print(variable_names)

            start_from_epoch = 0
            if self.continue_from_epoch != -1:
                start_from_epoch = self.continue_from_epoch
                # checkpoint = "{}train_saved_model_{}_{}.ckpt".format(self.saved_models_filepath, self.experiment_name, self.continue_from_epoch)
                checkpoint = self.restore_path
                variables_to_restore = []
                # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                #     variables_to_restore.append(var)
                for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='generator'):
                    variables_to_restore.append(var)

                tf.logging.info('Fine-tuning from %s' % checkpoint)

                fine_tune = slim.assign_from_checkpoint_fn(
                    checkpoint,
                    variables_to_restore,
                    ignore_missing_vars=True)
                fine_tune(sess)

            self.iter_done = 0
            self.disc_iter = 1
            self.gen_iter = 1
            best_d_val_loss = np.inf

            ### z preprocess
            self.spherical_interpolation = False
            if self.spherical_interpolation:
                self.z_vectors = interpolations.create_mine_grid(rows=1, cols=self.num_generations, dim=self.z_dim,
                                                                 space=3, anchors=None, spherical=True, gaussian=True)
                self.z_vectors_2 = interpolations.create_mine_grid(rows=1, cols=self.num_generations, dim=self.z_dim,
                                                                   space=3, anchors=None, spherical=True, gaussian=True)

                self.z_2d_vectors = interpolations.create_mine_grid(rows=self.num_generations,
                                                                    cols=self.num_generations,
                                                                    dim=100, space=3, anchors=None,
                                                                    spherical=True, gaussian=True)
                self.z_2d_vectors_2 = interpolations.create_mine_grid(rows=self.num_generations,
                                                                      cols=self.num_generations,
                                                                      dim=100, space=3, anchors=None,
                                                                      spherical=True, gaussian=True)


            else:
                self.z_vectors = np.random.normal(size=(self.num_generations, self.z_dim))
                self.z_vectors_2 = np.random.normal(size=(self.num_generations, self.z_dim))

                self.z_2d_vectors = np.random.normal(size=(self.num_generations, self.z_dim))
                self.z_2d_vectors_2 = np.random.normal(size=(self.num_generations, self.z_dim))

            self.support_vector = [x for x in range(10)]

            ### training with train set with parameter update, validation without training, epoch

            image_name = "z2vae{}_z2{}_g{}_d{}_kl{}_cla{}_fzl{}_reconsB{}_matchingG{}_matchingD{}_sim{}_Net_batchsize{}_classencodedim{}_imgsize{}".format(
                self.is_z2_vae, self.is_z2, self.loss_G, self.loss_D, self.loss_KL, self.loss_CLA, self.loss_FSL,
                self.loss_recons_B, self.loss_matching_G, self.loss_matching_D, self.loss_sim, self.batch_size,
                self.z_dim, self.image_width)

            if int(self.continue_from_epoch) > 900:
                print('sampling')
                # print('starting sampling')
                similarities_list = []
                f_encode_z_list = []
                matching_feature_list = []
                with tqdm.tqdm(total=self.total_gen_batches) as pbar_samp:
                    for i in range(self.total_gen_batches):
                        x_test_i_selected_classes, x_test_j, y_test_i_selected_classes, y_test_j, y_global_test_i_selected_classes, y_global_test_j = self.data.get_test_batch()
                        # np.random.seed(i)
                        # self.z_vectors = np.random.normal(size=(self.num_generations, self.z_dim))
                        # self.z_vectors_2 = np.random.normal(size=(self.num_generations, self.z_dim))
                        if i == 0:
                            for j in range(1):
                                before_sample = time.time()
                                x_test_i = x_test_i_selected_classes[:, :, j, :, :, :]
                                y_test_i = y_test_i_selected_classes[:, :, j, :]
                                y_global_test_i = y_global_test_i_selected_classes[:, :, j, :]

                                support_index = int(np.random.choice(self.data.support_number, size=1))
                                x_test_j_selected = x_test_j[:, :, support_index, :, :, :]
                                y_test_j_selected = y_test_j[:, :, support_index, :]
                                y_global_test_j_selected = y_global_test_j[:, :, support_index, :]

                                _, _, _ = sample_generator(num_generations=self.num_generations,
                                                           sess=sess,
                                                           same_images=self.same_images,
                                                           input_a=self.input_x_i,
                                                           input_b=self.input_x_j,
                                                           input_y_i=self.input_y_i,
                                                           input_y_j=self.input_y_j,
                                                           input_global_y_i=self.input_global_y_i,
                                                           input_global_y_j=self.input_global_y_j,
                                                           classes=self.classes,
                                                           classes_selected=self.selected_classes,
                                                           number_support=self.support_number,
                                                           z_input=self.z_input,
                                                           z_input_2=self.z_input_2,
                                                           # selected_global_x_j = self.input_x_j_selected,
                                                           # selected_global_y_j=self.input_global_y_j_selected,

                                                           # conditional_inputs=x_test_i,
                                                           # y_input_i = y_test_i,
                                                           conditional_inputs=x_test_j_selected,
                                                           y_input_i=y_test_j_selected,

                                                           support_input=x_test_j,
                                                           y_input_j=y_test_j,
                                                           y_global_input_i=y_global_test_i,
                                                           y_global_input_j=y_global_test_j,
                                                           classes_number=self.data.training_classes,
                                                           selected_classes=self.data.selected_classes,
                                                           support_number=self.data.support_number,
                                                           # input_global_x_j_selected = x_test_j_selected,
                                                           # input_global_y_j_selected = y_global_test_j_selected,
                                                           z_vectors=self.z_vectors,
                                                           z_vectors_2=self.z_vectors_2,
                                                           data=self.data,
                                                           batch_size=self.batch_size,
                                                           file_name="{}/{}_{}_{}.png".format(self.save_image_path,
                                                                                              image_name,
                                                                                              self.continue_from_epoch,
                                                                                              i),
                                                           dropout_rate=self.dropout_rate,
                                                           dropout_rate_value=self.dropout_rate_value,
                                                           training_phase=self.training_phase,
                                                           z1z2_training=self.z1z2_training,
                                                           is_training=False,
                                                           training_z1z2=False)
                                # if (f_encode_z[0]).all==(f_encode_z[1]).all:
                                #     print('same value in batchsize')
                                # csv_file_name = self.save_image_path
                                # np.savetxt(csv_file_name +'f1.txt'.format(i), f_encode_z[0], delimiter='\t', newline='\r\n')
                                # np.savetxt(csv_file_name +'f2.txt'.format(i), f_encode_z[1], delimiter='\t', newline='\r\n')

                                # print('hehre',np.shape(f_encode_z))
                                # np.savetxt(csv_file_name +'f_{}.txt'.format(i), f_encode_z, delimiter='\t', newline='\r\n')
                                # similarities_list.append(similarities)
                                # f_encode_z_list.append(f_encode_z)
                                # matching_feature_list.append(matching_feature)

                                after_sample = time.time()
                                # print('time for sampling', after_sample - before_sample)
                            pbar_samp.update(1)
                    # similarities_list1 = np.stack(similarities_list,axis=0)
                    # f_encode_z_list1 = np.stack(f_encode_z_list,axis=0)
                    # matching_feature_list1 = np.stack(matching_feature_list,axis=0)
                    # print(np.shape(similarities_list1),np.shape(f_encode_z_list1),np.shape(matching_feature_list1))
                    # similarities_list2 = np.reshape(similarities_list1,[np.shape(similarities_list1)[0]*np.shape(similarities_list1)[1],np.shape(similarities_list1)[2]])
                    # matching_feature_list2 = np.reshape(matching_feature_list1,[np.shape(matching_feature_list1)[0]*np.shape(matching_feature_list1)[1],np.shape(matching_feature_list1)[2]])
                    # f_encode_z_list2 = np.reshape(f_encode_z_list1,[np.shape(f_encode_z_list1)[0]*np.shape(f_encode_z_list1)[1],np.shape(f_encode_z_list1)[2]])
                    # np.savetxt(self.save_image_path +'z.txt', f_encode_z_list2, delimiter='\t', newline='\r\n')
                    # np.savetxt(self.save_image_path +'x.txt', matching_feature_list2, delimiter='\t', newline='\r\n')
                    # np.savetxt(self.save_image_path +'sim.txt', similarities_list2, delimiter='\t', newline='\r\n')

            with tqdm.tqdm(total=self.total_epochs - start_from_epoch) as pbar_e:
                for e in range(start_from_epoch, self.total_epochs):
                    train_g_loss = []
                    val_g_loss = []
                    train_d_loss = []
                    val_d_loss = []

                    train_classification_loss = []
                    val_classification_loss = []
                    train_fzl_classification_loss = []
                    val_fzl_classification_loss = []

                    ### total trianing batches
                    with tqdm.tqdm(total=self.total_train_batches) as pbar_train:
                        for iter in range(self.total_train_batches):
                            # before_classification = time.time()
                            for z1z2training in range(self.strategy):
                                if z1z2training == 0:
                                    z1z2_training = True
                                else:
                                    z1z2_training = False

                                if z1z2training < -1:
                                    for n in range(self.gen_iter):
                                        x_train_i_selected_classes, x_train_j, y_train_i_selected_classes, y_train_j, y_global_train_i_selected_classes, y_global_train_j = self.data.get_train_batch()
                                        # print('time for constructing batch:',time.time() - time_6)
                                        x_val_i_selected_classes, x_valid_j, y_valid_i_selected_classes, y_valid_j, y_global_val_i_selected_classes, y_global_val_j = self.data.get_val_batch()
                                        for i in range(1):
                                            x_train_i = x_train_i_selected_classes[:, :, i, :, :, :]
                                            y_train_i = y_train_i_selected_classes[:, :, i, :]
                                            y_global_train_i = y_global_train_i_selected_classes[:, :, i, :]

                                            x_valid_i = x_val_i_selected_classes[:, :, i, :, :, :]
                                            y_valid_i = y_valid_i_selected_classes[:, :, i, :]
                                            y_global_val_i = y_global_val_i_selected_classes[:, :, i, :]

                                            support_index = int(np.random.choice(self.data.support_number, size=1))
                                            x_train_j_selected = x_train_j[:, :,
                                                                 self.data.support_number * i + support_index, :, :, :]
                                            y_train_j_selected = y_train_j[:, :,
                                                                 self.data.support_number * i + support_index, :]
                                            y_global_train_j_selected = y_global_train_j[:, :,
                                                                        self.data.support_number * i + support_index, :]
                                            x_valid_j_selected = x_valid_j[:, :,
                                                                 self.data.support_number * i + support_index, :, :, :]
                                            y_global_val_j_selected = y_global_val_j[:, :,
                                                                      self.data.support_number * i + support_index, :]
                                            y_valid_j_selected = y_valid_j[:, :,
                                                                 self.data.support_number * i + support_index, :]

                                            _, g_train_loss_value, train_summaries = sess.run(
                                                [self.graph_ops["g_opt_op"], self.losses["g_losses"],
                                                 self.summary],
                                                feed_dict={
                                                    self.input_x_i: x_train_i,
                                                    self.input_y_i: y_train_i,
                                                    self.input_global_y_i: y_global_train_i,
                                                    # self.input_x_i: x_train_j_selected,
                                                    # self.input_y_i: y_train_j_selected,
                                                    # self.input_global_y_i: y_global_train_j_selected,
                                                    self.input_x_j: x_train_j,
                                                    self.input_x_j_selected: x_train_j_selected,
                                                    self.input_y_j: y_train_j,
                                                    self.input_y_j: y_train_j,
                                                    self.input_global_y_j: y_global_train_j,
                                                    self.input_global_y_j_selected: y_global_train_j_selected,
                                                    self.selected_classes: self.data.selected_classes,
                                                    self.support_number: self.data.support_number,
                                                    self.dropout_rate: self.dropout_rate_value,
                                                    self.training_phase: True, self.random_rotate: True,
                                                    self.z1z2_training: z1z2_training})
                                            train_g_loss.append(g_train_loss_value)
                                            if isNaN(g_train_loss_value):
                                                raise ValueError

                                            if iter % 50 == 0:
                                                g_val_loss_value, val_summaries = sess.run(
                                                    [self.losses["g_losses"], self.summary],
                                                    feed_dict={
                                                        self.input_x_i: x_valid_i,
                                                        self.input_y_i: y_valid_i,
                                                        self.input_global_y_i: y_global_val_i,
                                                        # self.input_x_i: x_valid_j_selected,
                                                        # self.input_y_i: y_valid_j_selected,
                                                        # self.input_global_y_i: y_global_val_j_selected,
                                                        self.input_x_j: x_valid_j,
                                                        self.input_x_j_selected: x_valid_j_selected,
                                                        self.input_y_j: y_valid_j,
                                                        self.input_global_y_j: y_global_val_j,
                                                        self.input_global_y_j_selected: y_global_val_j_selected,
                                                        self.selected_classes: self.data.selected_classes,
                                                        self.support_number: self.data.support_number,
                                                        self.dropout_rate: self.dropout_rate_value,
                                                        self.training_phase: False, self.random_rotate: False,
                                                        self.z1z2_training: z1z2_training})
                                                val_g_loss.append(g_val_loss_value)


                                else:
                                    #### data preparation
                                    x_train_i_selected_classes, x_train_j, y_train_i_selected_classes, y_train_j, y_global_train_i_selected_classes, y_global_train_j = self.data.get_train_batch()
                                    x_val_i_selected_classes, x_valid_j, y_valid_i_selected_classes, y_valid_j, y_global_val_i_selected_classes, y_global_val_j = self.data.get_val_batch()
                                    for i in range(1):
                                        x_train_i = x_train_i_selected_classes[:, :, i, :, :, :]
                                        y_train_i = y_train_i_selected_classes[:, :, i, :]
                                        y_global_train_i = y_global_train_i_selected_classes[:, :, i, :]

                                        x_valid_i = x_val_i_selected_classes[:, :, i, :, :, :]
                                        y_valid_i = y_valid_i_selected_classes[:, :, i, :]
                                        y_global_val_i = y_global_val_i_selected_classes[:, :, i, :]

                                        support_index = int(np.random.choice(self.data.support_number, size=1))
                                        x_train_j_selected = x_train_j[:, :,
                                                             self.data.support_number * i + support_index, :, :, :]
                                        y_train_j_selected = y_train_j[:, :,
                                                             self.data.support_number * i + support_index, :]
                                        y_global_train_j_selected = y_global_train_j[:, :,
                                                                    self.data.support_number * i + support_index, :]
                                        x_valid_j_selected = x_valid_j[:, :,
                                                             self.data.support_number * i + support_index, :, :, :]
                                        y_global_val_j_selected = y_global_val_j[:, :,
                                                                  self.data.support_number * i + support_index, :]
                                        y_valid_j_selected = y_valid_j[:, :,
                                                             self.data.support_number * i + support_index, :]
                                        for n in range(self.disc_iter):
                                            _, d_train_loss_value = sess.run(
                                                [self.graph_ops["d_opt_op"], self.losses["d_losses"]],
                                                feed_dict={
                                                    self.input_x_i: x_train_i,
                                                    self.input_global_y_i: y_global_train_i,
                                                    # self.input_x_i: x_train_j_selected,
                                                    # self.input_global_y_i: y_global_train_j_selected,
                                                    self.input_x_j: x_train_j,
                                                    self.input_x_j_selected: x_train_j_selected,
                                                    self.input_global_y_j: y_global_train_j,
                                                    self.input_global_y_j_selected: y_global_train_j_selected,
                                                    self.selected_classes: self.data.selected_classes,
                                                    self.support_number: self.data.support_number,
                                                    self.classes: self.data.training_classes,
                                                    self.dropout_rate: self.dropout_rate_value,
                                                    self.training_phase: True, self.random_rotate: True,
                                                    self.z1z2_training: z1z2_training})
                                            train_d_loss.append(d_train_loss_value)

                                            if iter % 50 == 0:
                                                d_val_loss_value = sess.run(
                                                    self.losses["d_losses"],
                                                    feed_dict={
                                                        self.input_x_i: x_valid_i,
                                                        self.input_global_y_i: y_global_val_i,
                                                        self.input_x_j: x_valid_j,
                                                        self.input_x_j_selected: x_valid_j_selected,
                                                        self.input_global_y_j_selected: y_global_val_j_selected,
                                                        self.input_global_y_j: y_global_val_j,
                                                        self.selected_classes: self.data.selected_classes,
                                                        self.support_number: self.data.support_number,
                                                        self.classes: self.data.training_classes,
                                                        self.dropout_rate: self.dropout_rate_value,
                                                        self.training_phase: False, self.random_rotate: False,
                                                        self.z1z2_training: z1z2_training})
                                            val_d_loss.append(d_val_loss_value)

                                        for n in range(self.gen_iter):
                                            _, g_train_loss_value, train_summaries = sess.run(
                                                [self.graph_ops["g_opt_op"], self.losses["g_losses"],
                                                 self.summary],
                                                feed_dict={
                                                    self.input_x_i: x_train_i,
                                                    self.input_y_i: y_train_i,
                                                    self.input_global_y_i: y_global_train_i,
                                                    # self.input_x_i: x_train_j_selected,
                                                    # self.input_y_i: y_train_j_selected,
                                                    # self.input_global_y_i: y_global_train_j_selected,
                                                    self.input_x_j: x_train_j,
                                                    self.input_x_j_selected: x_train_j_selected,
                                                    self.input_y_j: y_train_j,
                                                    self.input_y_j: y_train_j,
                                                    self.input_global_y_j: y_global_train_j,
                                                    self.input_global_y_j_selected: y_global_train_j_selected,
                                                    self.selected_classes: self.data.selected_classes,
                                                    self.support_number: self.data.support_number,
                                                    self.dropout_rate: self.dropout_rate_value,
                                                    self.training_phase: True, self.random_rotate: True,
                                                    self.z1z2_training: z1z2_training})
                                            train_g_loss.append(g_train_loss_value)
                                            if isNaN(g_train_loss_value):
                                                raise ValueError

                                            if iter % 50 == 0:
                                                g_val_loss_value, val_summaries = sess.run(
                                                    [self.losses["g_losses"], self.summary],
                                                    feed_dict={
                                                        self.input_x_i: x_valid_i,
                                                        self.input_y_i: y_valid_i,
                                                        self.input_global_y_i: y_global_val_i,
                                                        # self.input_x_i: x_valid_j_selected,
                                                        # self.input_y_i: y_valid_j_selected,
                                                        # self.input_global_y_i: y_global_val_j_selected,
                                                        self.input_x_j: x_valid_j,
                                                        self.input_x_j_selected: x_valid_j_selected,
                                                        self.input_y_j: y_valid_j,
                                                        self.input_global_y_j: y_global_val_j,
                                                        self.input_global_y_j_selected: y_global_val_j_selected,
                                                        self.selected_classes: self.data.selected_classes,
                                                        self.support_number: self.data.support_number,
                                                        self.dropout_rate: self.dropout_rate_value,
                                                        self.training_phase: False, self.random_rotate: False,
                                                        self.z1z2_training: z1z2_training})
                                                val_g_loss.append(g_val_loss_value)
                            # cur_sample += 1

                            # if iter % (self.tensorboard_update_interval) == 0:
                            self.train_writer.add_summary(train_summaries, global_step=self.iter_done)
                            self.validation_writer.add_summary(val_summaries, global_step=self.iter_done)
                            # after_generation = time.time()
                            # print('time for generation', after_generation - after_fzl_classification)

                            if iter == self.total_train_batches - 1:
                                _, _, _ = sample_generator(num_generations=self.num_generations,
                                                           sess=sess,
                                                           same_images=self.same_images,
                                                           input_a=self.input_x_i,
                                                           input_b=self.input_x_j,
                                                           input_y_i=self.input_y_i,
                                                           input_y_j=self.input_y_j,
                                                           input_global_y_i=self.input_global_y_i,
                                                           input_global_y_j=self.input_global_y_j,
                                                           classes=self.classes,
                                                           classes_selected=self.selected_classes,
                                                           number_support=self.support_number,
                                                           # selected_global_x_j = self.input_x_j_selected,
                                                           # selected_global_y_j=self.input_global_y_j_selected,
                                                           z_vectors=self.z_vectors,
                                                           z_vectors_2=self.z_vectors_2,
                                                           conditional_inputs=x_train_i,
                                                           y_global_input_i=y_global_train_i,
                                                           y_input_i=y_train_i,
                                                           # conditional_inputs=x_train_j_selected,
                                                           # y_global_input_i = y_global_train_j_selected,
                                                           # y_input_i = y_train_j_selected,
                                                           support_input=x_train_j,
                                                           y_input_j=y_train_j,
                                                           y_global_input_j=y_global_train_j,
                                                           classes_number=self.data.training_classes,
                                                           selected_classes=self.data.selected_classes,
                                                           support_number=self.data.support_number,
                                                           z_input=self.z_input,
                                                           z_input_2=self.z_input_2,
                                                           data=self.data,
                                                           batch_size=self.batch_size,
                                                           # input_global_x_j_selected = x_train_j_selected,
                                                           # input_global_y_j_selected = y_global_train_j_selected,

                                                           file_name="{}/train_z_variations_{}_{}_{}.png".format(
                                                               self.save_image_path,
                                                               image_name,
                                                               e, i),
                                                           dropout_rate=self.dropout_rate,
                                                           dropout_rate_value=self.dropout_rate_value,
                                                           training_phase=self.training_phase,
                                                           z1z2_training=self.z1z2_training,
                                                           is_training=False,
                                                           training_z1z2=True)
                                # print('time for sampling:', time.time() - after_generation)

                            self.iter_done = self.iter_done + 1
                            if iter % 50 == 0:
                                if self.loss_FSL > 0:
                                    iter_out = "{}_train_d_loss: {}, train_g_loss: {}, \
                                           val_d_loss: {}, val_g_loss: {}".format(self.iter_done,
                                                                                  d_train_loss_value,
                                                                                  g_train_loss_value,
                                                                                  d_val_loss_value,
                                                                                  g_val_loss_value,
                                                                                  )
                                else:
                                    iter_out = "{}_train_d_loss: {}, train_g_loss: {}, \
                                               val_d_loss: {}, val_g_loss: {}".format(self.iter_done,
                                                                                      d_train_loss_value,
                                                                                      g_train_loss_value,
                                                                                      d_val_loss_value,
                                                                                      g_val_loss_value,
                                                                                      )
                            else:
                                if self.loss_FSL > 0:
                                    iter_out = "{}_train_d_loss: {}, train_g_loss: {}".format(self.iter_done,
                                                                                              d_train_loss_value,
                                                                                              g_train_loss_value,
                                                                                              )
                                else:
                                    iter_out = "{}_train_d_loss: {}, train_g_loss: {}".format(self.iter_done,
                                                                                              d_train_loss_value,
                                                                                              g_train_loss_value
                                                                                              )

                            pbar_train.set_description(iter_out)
                            pbar_train.update(1)

                    ### each epoch information output
                    total_d_train_loss_mean = np.mean(train_d_loss)
                    total_d_train_loss_std = np.std(train_d_loss)
                    total_g_train_loss_mean = np.mean(train_g_loss)
                    total_g_train_loss_std = np.std(train_g_loss)
                    print(
                        "Epoch {}: d_train_loss_mean: {}, d_train_loss_std: {},\
                        g_train_loss_mean: {}, g_train_loss_std: {}"
                            .format(e, total_d_train_loss_mean,
                                    total_d_train_loss_std,
                                    total_g_train_loss_mean,
                                    total_g_train_loss_std))

                    total_d_val_loss_mean = np.mean(val_d_loss)
                    total_d_val_loss_std = np.std(val_d_loss)
                    total_g_val_loss_mean = np.mean(val_g_loss)
                    total_g_val_loss_std = np.std(val_g_loss)

                    print(
                        "Epoch {}: d_val_loss_mean: {}, d_val_loss_std: {},\
                        g_val_loss_mean: {}, g_val_loss_std: {}"
                            .format(e, total_d_val_loss_mean,
                                    total_d_val_loss_std,
                                    total_g_val_loss_mean,
                                    total_g_val_loss_std))

                    # print('starting sampling')
                    with tqdm.tqdm(total=self.total_gen_batches) as pbar_samp:
                        np.random.seed(0)
                        for i in range(self.total_gen_batches):
                            x_test_i_selected_classes, x_test_j, y_test_i_selected_classes, y_test_j, y_global_test_i_selected_classes, y_global_test_j = self.data.get_test_batch()
                            if i == 0:
                                for j in range(1):
                                    before_sample = time.time()
                                    x_test_i = x_test_i_selected_classes[:, :, j, :, :, :]
                                    y_test_i = y_test_i_selected_classes[:, :, j, :]
                                    y_global_test_i = y_global_test_i_selected_classes[:, :, j, :]

                                    support_index = int(np.random.choice(self.data.support_number, size=1))
                                    x_test_j_selected = x_test_j[:, :, support_index, :, :, :]
                                    y_global_test_j_selected = y_global_test_j[:, :, support_index, :]

                                    _, _, _ = sample_generator(num_generations=self.num_generations,
                                                               sess=sess,
                                                               same_images=self.same_images,
                                                               input_a=self.input_x_i,
                                                               input_b=self.input_x_j,
                                                               input_y_i=self.input_y_i,
                                                               input_y_j=self.input_y_j,
                                                               input_global_y_i=self.input_global_y_i,
                                                               input_global_y_j=self.input_global_y_j,
                                                               classes=self.classes,
                                                               classes_selected=self.selected_classes,
                                                               number_support=self.support_number,
                                                               # selected_global_x_j = self.input_x_j_selected,
                                                               # selected_global_y_j=self.input_global_y_j_selected,

                                                               z_vectors=self.z_vectors,
                                                               z_vectors_2=self.z_vectors_2,
                                                               conditional_inputs=x_test_i,
                                                               y_input_i=y_test_i,
                                                               y_global_input_i=y_global_test_i,
                                                               # conditional_inputs=x_test_j_selected,
                                                               # y_input_i = y_test_j_selected,
                                                               # y_global_input_i = y_global_test_j_selected,

                                                               support_input=x_test_j,
                                                               y_input_j=y_test_j,
                                                               y_global_input_j=y_global_test_j,
                                                               classes_number=self.data.training_classes,
                                                               selected_classes=self.data.selected_classes,
                                                               support_number=self.data.support_number,
                                                               z_input=self.z_input,
                                                               z_input_2=self.z_input_2,
                                                               data=self.data,
                                                               batch_size=self.batch_size,
                                                               # input_global_x_j_selected = x_train_j_selected,
                                                               # input_global_y_j_selected = y_global_train_j_selected,
                                                               file_name="{}/test_z_variations_{}_{}_{}.png".format(
                                                                   self.save_image_path,
                                                                   image_name,
                                                                   e, j),

                                                               dropout_rate=self.dropout_rate,
                                                               dropout_rate_value=self.dropout_rate_value,
                                                               training_phase=self.training_phase,
                                                               z1z2_training=self.z1z2_training,
                                                               is_training=False,
                                                               training_z1z2=False)
                                    after_sample = time.time()
                                # print('time for sampling', after_sample - before_sample)
                            pbar_samp.update(1)

                    ###tf.train.Saver().save(sess)
                    if e % 5 == 0 or e < 5:
                        train_save_path = self.train_saver.save(sess,
                                                                "{}/train_LOSS_z2vae{}_z2{}_g{}_d{}_kl{}_cla{}_fzl_cla{}_reconsB{}_matchingG{}_matchingD{}_sim{}_Net_batchsize{}_classencodedim{}_imgsize{}_epoch{}.ckpt".format(
                                                                    self.saved_models_filepath, self.is_z2_vae,
                                                                    self.is_z2, self.loss_G, self.loss_D, self.loss_KL,
                                                                    self.loss_CLA, self.loss_FSL, self.loss_recons_B,
                                                                    self.loss_matching_G, self.loss_matching_D,
                                                                    self.loss_sim, self.batch_size, self.z_dim,
                                                                    self.image_width, e))

                    ### validation selection, according to the loss of discriminator.
                    if total_d_val_loss_mean < best_d_val_loss:
                        best_d_val_loss = total_d_val_loss_mean
                        val_save_path = self.train_saver.save(sess,
                                                              "{}/valid_LOSS_z2vae{}_z2{}_g{}_d{}_kl{}_cla{}_fzl_cla{}_reconsB{}_matchingG{}_matchingD{}_sim{}_Net_batchsize{}_classencodedim{}_imgsize{}_epoch{}.ckpt".format(
                                                                  self.saved_models_filepath, self.is_z2_vae,
                                                                  self.is_z2, self.loss_G, self.loss_D, self.loss_KL,
                                                                  self.loss_CLA, self.loss_FSL, self.loss_recons_B,
                                                                  self.loss_matching_G, self.loss_matching_D,
                                                                  self.loss_sim, self.batch_size, self.z_dim,
                                                                  self.image_width, e))

                        print("Saved current best val model at", val_save_path)

                    # save_statistics(self.log_path, [e, total_d_train_loss_mean, total_d_val_loss_mean,
                    #                             total_d_train_loss_std, total_d_val_loss_std,
                    #                             total_g_train_loss_mean, total_g_val_loss_mean,
                    #                             total_g_train_loss_std, total_g_val_loss_std])
                    pbar_e.update(1)

