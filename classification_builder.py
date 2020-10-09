import utils.interpolations as interpolations
import tqdm
from utils.storage import *
from tensorflow.contrib import slim

from dagan_networks_wgan_with_matchingclassifier import *
from utils.sampling_with_matchingclassifier import *
import sys

# sys.path.append("./models/research/slim")
# from nets.mobilenet import mobilenet_v2
slim = tf.contrib.slim

from densenet_classifier import densenet_classifier

# pretrained_resnet = './models/pretrained_model/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
pretrained_resnet = './models/pretrained_model/resnet_v1_50.ckpt'
checkpoint_exclude_scopes = 'resnet_v1_50/logits'
model_name = 'resnet_v1_50'
checkpoint_path = './models/pretrained_model/resnet_v1_50.ckpt'
checkpoint_exclude_scopes = 'resnet_v1_50/logits'


#### for mobilenet
class ExperimentBuilder(object):
    def __init__(self, parser, data):
        tf.reset_default_graph()
        args = parser.parse_args()
        self.continue_from_epoch = args.continue_from_epoch
        self.experiment_name = args.experiment_title
        self.saved_models_filepath, self.log_path, self.save_image_path = build_experiment_folder(self.experiment_name)
        self.num_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        gen_depth_per_layer = args.generator_inner_layers
        discr_depth_per_layer = args.discriminator_inner_layers
        self.z_dim = args.z_dim
        self.num_generations = args.num_generations
        self.dropout_rate_value = args.dropout_rate_value
        self.data = data
        self.reverse_channels = False
        self.support_number = args.support_number
        self.classification_total_epoch = args.classification_total_epoch
        image_channel = data.image_channel
        self.use_wide_connections = args.use_wide_connections
        self.pretrain = args.pretrain

        generator_layers = [64, 64, 128, 128]
        self.discriminator_layers = [64, 64, 128, 128]

        gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer]
        self.discr_inner_layers = [discr_depth_per_layer, discr_depth_per_layer, discr_depth_per_layer,
                                   discr_depth_per_layer]
        generator_layer_padding = ["SAME", "SAME", "SAME", "SAME"]

        image_height = self.data.image_width
        image_width = self.data.image_width
        image_channels = self.data.image_channel

        self.classes = tf.placeholder(tf.int32)
        self.input_x_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size*self.data.selected_classes, image_height, image_width,
                                                     image_channels], 'inputs-1')

        self.input_y = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size*self.data.selected_classes, self.data.selected_classes],
                                      'y_inputs-1')


        self.input_x_j = tf.placeholder(tf.float32,
                                        [self.num_gpus, self.batch_size, self.support_number, image_height, image_width,
                                         image_channels], 'inputs-2-same-class')

        self.z_input = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input')
        self.z_input_2 = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input_2')

        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.random_rotate = tf.placeholder(tf.bool, name='rotation-flag')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')

        self.matching = args.matching
        self.fce = args.fce
        self.full_context_unroll_k = args.full_context_unroll_k
        self.average_per_class_embeddings = args.average_per_class_embeddings

        self.total_train_batches = data.training_data_size / (self.batch_size * self.num_gpus)
        self.total_val_batches = data.validation_data_size / (self.batch_size * self.num_gpus)
        self.total_test_batches = 5*545 / (self.batch_size * self.num_gpus)
        self.total_gen_batches = data.generation_data_size / (self.batch_size * self.num_gpus)
        self.spherical_interpolation = True

        self.tensorboard_update_interval = int(self.total_test_batches / 10 / self.num_gpus)


        classifier = densenet_classifier(input_x_i=self.input_x_i, input_y=self.input_y,
                                         classes=self.data.selected_classes,
                                         batch_size=self.batch_size, layer_sizes=self.discriminator_layers,
                                         inner_layers=self.discr_inner_layers, num_gpus=self.num_gpus,
                                         use_wide_connections=self.use_wide_connections,
                                         is_training=self.training_phase, augment=self.random_rotate,
                                         dropout_rate=self.dropout_rate)
        print('classes',self.data.selected_classes)


        self.summary, self.losses, self.accuracy, self.graph_ops = classifier.init_train()
        self.init = tf.global_variables_initializer()

    def run_experiment(self):

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(self.init)
            # self.train_writer = tf.summary.FileWriter("{}/train_classification_logs/".format(self.log_path),
            #                                           graph=tf.get_default_graph())
            # self.valid_writer = tf.summary.FileWriter("{}/validation_classification_logs/".format(self.log_path),
            #                                                graph=tf.get_default_graph())
            print('Load parameters from basemodel')
            variables = tf.global_variables()
            vars_restore = [var for var in variables
                            if not "Momentum" in var.name and
                            not "global_step" in var.name and var.name.split('/')[0]!='logits']


            # print('load weights',vars_restore)


            ### remove fc weights to load
            # tf.contrib.framework.get_variables("logits/fc/weights")
            # print('network variable', vars_restore)
            saver_restore = tf.train.Saver(vars_restore, max_to_keep=10000)
            checkpoint = './pretrained-resnet18model/model.ckpt'
            # reader = tf.train.NewCheckpointReader(checkpoint)
            # var_to_shape_map = reader.get_variable_to_shape_map()
            # for key in var_to_shape_map:
            #     print("store model name: ", key)

            fine_tune = slim.assign_from_checkpoint_fn(
                checkpoint,
                vars_restore,
                ignore_missing_vars=True)
            fine_tune(sess)



            # saver_restore.restore(sess, checkpoint)
            self.saver = tf.train.Saver()

            self.iter_done = 0
            best_d_val_loss = np.inf
            lowest_d_val_accuracy = 0
            with tqdm.tqdm(total=self.classification_total_epoch) as pbar_e:
                for e in range(self.classification_total_epoch):
                    train_loss = []
                    train_acc = []
                    test_loss = []
                    test_acc = []
                    with tqdm.tqdm(total=self.total_test_batches) as pbar_samp:

                        for iter in range(int(self.total_test_batches)):
                            if self.pretrain > 0:
                                x_test_classification_i, x_train_classification_i, _, _, y_test, y_train  = self.data.get_train_batch()
                            else:
                                x_test_classification_i, x_train_classification_i, _, _, y_test, y_train  = self.data.get_test_batch()

                            x_train_i = np.reshape(x_train_classification_i,
                                                                  [x_train_classification_i.shape[0],
                                                                   x_train_classification_i.shape[1] *
                                                                   x_train_classification_i.shape[2], \
                                                                   x_train_classification_i.shape[3],
                                                                   x_train_classification_i.shape[4],
                                                                   x_train_classification_i.shape[5]])

                            x_test_i = np.reshape(x_test_classification_i,
                                                 [x_test_classification_i.shape[0],
                                                  x_test_classification_i.shape[1] *
                                                  x_test_classification_i.shape[2], \
                                                  x_test_classification_i.shape[3],
                                                  x_test_classification_i.shape[4],
                                                  x_test_classification_i.shape[5]])


                            # x_train_i = np.reshape(x_train_i,
                            #                        [x_train_i.shape[0] * x_train_i.shape[1], \
                            #                         x_train_i.shape[2], x_train_i.shape[3], x_train_i.shape[4]])
                            #
                            # x_test_i = np.reshape(x_test_i,
                            #                        [x_test_i.shape[0] * x_test_i.shape[1], \
                            #                         x_test_i.shape[2], x_test_i.shape[3], x_test_i.shape[4]])




                            y_train = np.reshape(y_train, [y_train.shape[0], y_train.shape[1] * y_train.shape[2],
                                                           y_train.shape[3]])
                            y_test = np.reshape(y_test, [y_test.shape[0], y_test.shape[1] * y_test.shape[2], y_test.shape[3]])

                            _, train_loss_value, train_acc_value, train_summary = sess.run(
                                [self.graph_ops["loss_opt_op"], self.losses["losses"], self.accuracy, self.summary],
                                feed_dict={self.input_x_i: x_train_i,
                                           self.input_y: y_train,
                                           self.dropout_rate: self.dropout_rate_value,
                                           self.training_phase: True, self.random_rotate: True})

                            test_loss_value, test_acc_value, test_summary = sess.run(
                                [self.losses["losses"], self.accuracy, self.summary],
                                feed_dict={self.input_x_i:x_test_i,
                                           self.input_y: y_test,
                                           self.dropout_rate: self.dropout_rate_value,
                                           self.training_phase: False, self.random_rotate: False})

                            train_loss.append(train_loss_value)
                            train_acc.append(train_acc_value)
                            test_loss.append(test_loss_value)
                            test_acc.append(test_acc_value)

                            # if iter % (self.tensorboard_update_interval) == 0:
                            #     self.train_writer.add_summary(train_summary, global_step=self.iter_done)
                            #     self.valid_writer.add_summary(test_summary, global_step=self.iter_done)

                        total_train_loss_mean = np.mean(train_loss)
                        total_train_accuracy_mean = np.mean(train_acc)
                        total_test_loss_mean = np.mean(test_loss)
                        total_test_accuracy_mean = np.mean(test_acc)
                        iter_out = "{},total_test_loss: {}, total_test_accuracy: {} , total_train_loss: {}, total_train_accuracy: {} ".format(iter, total_test_loss_mean,
                                                                                             total_test_accuracy_mean, total_train_loss_mean,
                                                                                             total_train_accuracy_mean, )
                        pbar_e.set_description(iter_out)
                        pbar_e.update(1)

                    # train_save_path = self.saver.save(sess, "{}/train_saved_model_{}_{}.ckpt".format(
                    #     self.saved_models_filepath,
                    #     self.experiment_name, e))

                    ### validation selection, according to the loss of discriminator.
                    model_name = 'Pretrain_on_Source_Domain'
                    if total_test_loss_mean < best_d_val_loss:
                        best_d_val_loss = total_test_loss_mean
                        val_save_path = self.saver.save(sess, "{}/{}_{}_{}.ckpt".format(
                            self.saved_models_filepath,
                            model_name, e, total_test_accuracy_mean))
                        print("Saved current best val model at", val_save_path)
                    if total_test_accuracy_mean > lowest_d_val_accuracy:
                        lowest_d_val_accuracy = total_test_accuracy_mean
                        val_save_path = self.saver.save(sess, "{}/{}_{}_{}.ckpt".format(
                            self.saved_models_filepath,
                            model_name, e, total_test_accuracy_mean))
                        print("Saved current best val model at", val_save_path)
                    pbar_e.update(1)


















