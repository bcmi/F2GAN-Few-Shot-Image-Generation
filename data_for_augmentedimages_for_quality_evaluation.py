import numpy as np

np.random.seed(2591)
import os
import cv2


# from data_preparation import one_channel_evaluation, three_channel_evaluation


class DAGANDataset(object):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        """
        :param batch_size: The batch size to use for the data loader
        :param last_training_class_index: The final index for the training set, used to restrict the training set
        if needed. E.g. if training set is 1200 classes and last_training_class_index=900 then only the first 900
        classes will be used
        :param reverse_channels: A boolean indicating whether we need to reverse the colour channels e.g. RGB to BGR
        :param num_of_gpus: Number of gpus to use for training
        :param gen_batches: How many batches to use from the validation set for the end of epoch generations
        """
        self.x_train, self.x_test, self.x_val = self.load_dataset(last_training_class_index)
        # (900, 20, 28, 28, 1)  (400, 20, 28, 28, 1)  (22, 20, 28, 28, 1)
        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.reverse_channels = reverse_channels
        self.test_samples_per_label = gen_batches
        self.support_number = support_number
        self.is_training = is_training
        self.general_classification_samples = general_classification_samples
        self.selected_classes = selected_classes
        self.image_size = image_size

        ### reptition choosen 32 classes from 22 categories, reptition choosen 1000 samples from each category
        ### selecting several categories from the validation set

        # self.choose_gen_labels = np.random.choice(self.x_val.shape[0], self.batch_size, replace=True)
        # self.choose_gen_samples = np.random.choice(len(self.x_val[0]), self.test_samples_per_label, replace=True)
        # self.x_gen = self.x_val[self.choose_gen_labels]
        # self.x_gen = self.x_gen[:, self.choose_gen_samples]
        # self.x_gen = np.reshape(self.x_gen, newshape=(self.x_gen.shape[0] * self.x_gen.shape[1],
        #                                         self.x_gen.shape[2], self.x_gen.shape[3], self.x_gen.shape[4]))
        # self.gen_batches = gen_batches

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.indexes = {"train": 0, "val": 0, "test": 0, "gen": 0}
        self.datasets = {"train": self.x_train,
                         "val": self.x_val,
                         "test": self.x_test}

        self.image_height = self.image_size
        self.image_width = self.image_size
        self.image_channel = self.x_train[0].shape[3]
        ## classes
        self.training_classes = self.x_train.shape[0]
        self.testing_classes = self.x_test.shape[0]
        self.val_classes = self.x_val.shape[0]
        ## classes * samples
        self.training_data_size = self.x_train.shape[0] * self.x_train[0].shape[0]
        self.testing_data_size = self.x_test.shape[0] * self.x_test[0].shape[0]

        ## gen_batches=10, how many batches used for generation
        self.validation_data_size = self.x_val.shape[0] * self.x_val[0].shape[0]
        self.generation_data_size = self.validation_data_size

    def load_dataset(self, last_training_class_index):
        """
        Loads the dataset into the data loader class. To be implemented in all classes that inherit
        DAGANImbalancedDataset
        :param last_training_class_index: last_training_class_index: The final index for the training set,
        used to restrict the training set if needed. E.g. if training set is 1200 classes and
        last_training_class_index=900 then only the first 900 classes will be used
        """
        raise NotImplementedError

    def preprocess_data(self, x):
        """
        Preprocesses data such that their values lie in the -1.0 to 1.0 range so that the tanh activation gen output
        can work properly
        :param x: A data batch to preprocess
        :return: A preprocessed data batch
        """
        x = 2 * x - 1
        if self.reverse_channels:
            reverse_photos = np.ones(shape=x.shape)
            for channel in range(x.shape[-1]):
                reverse_photos[:, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, channel]
            x = reverse_photos
        return x

    def reconstruct_original(self, x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        x = (x + 1) / 2
        return x

    def shuffle(self, x):
        """
        Shuffles the data batch along it's first axis
        :param x: A data batch
        :return: A shuffled data batch
        """
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        return x

    def get_total_batch_images(self, dataset_name, samples_number_each_category):
        categories = self.x_test.shape[0]
        # samples_index = np.random.choice(self.datasets[dataset_name].shape[1], size=samples_number_each_category, replace=True)
        total_samples = np.zeros(
            [categories * samples_number_each_category, self.image_height, self.image_height, self.image_channel])
        for i in range(categories):
            for j in range(samples_number_each_category):
                # print('here',samples_number_each_category*i+j)
                total_samples[samples_number_each_category * i + j] = self.resize(self.datasets[dataset_name][i][j])
        total_samples = total_samples * 255
        return total_samples

    def resize(self, image):
        # image = np.int(255*image)
        image = cv2.resize(image, (self.image_width, self.image_width), interpolation=cv2.INTER_LINEAR)
        if self.image_channel < 3:
            image = np.expand_dims(image, -1)
        return image

    def rgb2gray(self, rgb):
        image = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(image, axis=-1)
        return image

    def get_batch(self, dataset_name):
        if self.is_training > 0:
            classes = self.training_classes
        else:
            # classes = self.training_classes
            classes = self.testing_classes

        if self.support_number > 5:
            classes = self.training_classes

        x_input_batch_a = np.zeros(
            [self.batch_size, self.selected_classes, self.image_height, self.image_width,
             self.image_channel])
        y_input_batch_a = np.zeros([self.batch_size, self.selected_classes, self.selected_classes])
        y_global_input_batch_a = np.zeros([self.batch_size, self.selected_classes, classes])

        x_input_batch_b = np.zeros(
            [self.batch_size, self.selected_classes * self.support_number, self.image_height, self.image_width,
             self.image_channel])
        y_input_batch_b = np.zeros(
            [self.batch_size, self.selected_classes * self.support_number, self.selected_classes])
        y_global_input_batch_b = np.zeros([self.batch_size, self.selected_classes * self.support_number, classes])

        ##### training ot testing few-shot classifier
        # few-shot setting
        # x_input_batch_a is one samples from the n-way-k-shot
        # x_input_batch_b are N*K samples from the n-way-k-shot
        ##### testing general classifier
        # for n-way-1-shot matchingGAN, X_Bi can be selected from the X_Si
        # x_input_batch_a is
        # print('total',np.shape(self.datasets[dataset_name])) (1200, 20, 28, 28, 1)
        # xb_datasets = self.datasets[dataset_name][:, :1, :, :, :]
        # xs_datasets = self.datasets[dataset_name][:, 1:, :, :, :]
        if self.is_training > 0:
            for i in range(self.batch_size):
                choose_classes = np.random.choice(len(self.datasets[dataset_name]), size=self.selected_classes)
                # choose_classes = [(i*self.selected_classes+j) for j in range(self.selected_classes)]
                for j in range(self.selected_classes):
                    index = np.array([k for k in range(0, self.datasets[dataset_name][choose_classes[j]].shape[0])])
                    choose_samples = np.random.choice(index, size=self.support_number, replace=False)
                    x_input_batch_a[i, j, :, :, :] = self.resize(self.datasets[dataset_name][choose_classes[j]][0])
                    y_input_batch_a[i, j, j] = 1
                    y_global_input_batch_a[i, j, choose_classes[j]] = 1

                    for k in range(self.support_number):
                        x_input_batch_b[i, self.support_number * j + k, :, :, :] = \
                            self.resize(self.datasets[dataset_name][choose_classes[j]][choose_samples[k]])
                        y_input_batch_b[i, self.support_number * j + k, j] = 1
                        y_global_input_batch_b[i, self.support_number * j + k, choose_classes[j]] = 1

            for i in range(self.selected_classes):
                x_input_batch_a[:, i] = self.preprocess_data(x_input_batch_a[:, i])
            for j in range(self.selected_classes * self.support_number):
                x_input_batch_b[:, j] = self.preprocess_data(x_input_batch_b[:, j])
            return x_input_batch_a, x_input_batch_b, y_input_batch_a, y_input_batch_b, y_global_input_batch_a, y_global_input_batch_b

        else:

            #### for trained matchingGAN to generate fake images
            print('dataset name', dataset_name)
            print('total data', np.shape(self.datasets[dataset_name]))
            training_dataset = self.datasets[dataset_name][:, :self.general_classification_samples]
            #### testing for classifier
            testing_number = int(self.general_classification_samples * 0.4)
            testing_dataset = self.datasets[dataset_name][:, self.general_classification_samples:]
            print('training', np.shape(training_dataset))
            print('testing', np.shape(testing_dataset))

            self.training_data_size = len(training_dataset) * len(training_dataset[0])
            self.testing_data_size = len(testing_dataset) * len(testing_dataset[0])

            for i in range(self.batch_size):
                choose_classes = np.random.choice(len(training_dataset), size=self.selected_classes)
                for j in range(self.selected_classes):
                    choose_samples_a = np.random.choice(testing_dataset[choose_classes[j]].shape[0], size=1,
                                                        replace=False)
                    choose_samples_b = np.random.choice(training_dataset[choose_classes[j]].shape[0],
                                                        size=self.support_number, replace=False)

                    x_input_batch_a[i, j, :, :, :] = self.resize(
                        testing_dataset[choose_classes[j]][choose_samples_a[0]])
                    y_input_batch_a[i, j, j] = 1
                    y_global_input_batch_a[i, j, choose_classes[j]] = 1

                    for k in range(self.support_number):
                        x_input_batch_b[i, self.support_number * j + k, :, :, :] = self.resize(
                            training_dataset[choose_classes[j]][
                                choose_samples_b[k]])
                        y_input_batch_b[i, self.support_number * j + k, j] = 1
                        y_global_input_batch_b[i, self.support_number * j + k, choose_classes[j]] = 1

            for i in range(self.selected_classes):
                x_input_batch_a[:, i] = self.preprocess_data(x_input_batch_a[:, i])
            for j in range(self.selected_classes * self.support_number):
                x_input_batch_b[:, j] = self.preprocess_data(x_input_batch_b[:, j])
            return x_input_batch_a, x_input_batch_b, y_input_batch_a, y_input_batch_b, y_global_input_batch_a, y_global_input_batch_b

    def get_next_gen_batch(self):
        """
        Provides a batch that contains data to be used for generation
        :return: A data batch to use for generation
        """
        if self.indexes["gen"] >= self.batch_size * self.gen_batches:
            self.indexes["gen"] = 0
        x_input_batch_a = self.datasets["gen"][self.indexes["gen"]:self.indexes["gen"] + self.batch_size]
        self.indexes["gen"] += self.batch_size
        return self.preprocess_data(x_input_batch_a)

    def get_multi_batch(self, dataset_name):
        """
        Returns a batch to be used for training or evaluation for multi gpu training
        :param set_name: The name of the data-set to use e.g. "train", "test" etc
        :return: Two batches (i.e. x_i and x_j) of size [num_gpus, batch_size, im_height, im_width, im_channels). If
        the set is "gen" then we only return a single batch (i.e. x_i)
        """
        x_input_a_batch = []
        x_input_b_batch = []
        y_input_batch_a = []
        y_input_batch_b = []
        y_global_input_batch_a = []
        y_global_input_batch_b = []
        if dataset_name == "gen":
            x_input_a = self.get_next_gen_batch()
            for n_batch in range(self.num_of_gpus):
                x_input_a_batch.append(x_input_a)
            x_input_a_batch = np.array(x_input_a_batch)
            return x_input_a_batch
        else:
            for n_batch in range(self.num_of_gpus):
                x_input_a, x_input_b, y_input_a, y_input_b, y_global_input_a, y_global_input_b = self.get_batch(
                    dataset_name)
                x_input_a_batch.append(x_input_a)
                x_input_b_batch.append(x_input_b)
                y_input_batch_a.append(y_input_a)
                y_input_batch_b.append(y_input_b)
                y_global_input_batch_a.append(y_global_input_a)
                y_global_input_batch_b.append(y_global_input_b)

            x_input_a_batch = np.array(x_input_a_batch)
            x_input_b_batch = np.array(x_input_b_batch)
            y_input_batch_a = np.array(y_input_batch_a)
            y_input_batch_b = np.array(y_input_batch_b)
            y_global_input_batch_a = np.array(y_global_input_batch_a)
            y_global_input_batch_b = np.array(y_global_input_batch_b)
            return x_input_a_batch, x_input_b_batch, y_input_batch_a, y_input_batch_b, y_global_input_batch_a, y_global_input_batch_b

    def get_train_batch(self):
        """
        Provides a training batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for training
        """
        x_input_a, x_input_b, y_input_a, y_input_b, y_global_input_a, y_global_input_b = self.get_multi_batch("train")
        return x_input_a, x_input_b, y_input_a, y_input_b, y_global_input_a, y_global_input_b

    def get_test_batch(self):
        """
        Provides a test batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for evaluation
        """
        x_input_a, x_input_b, y_input_a, y_input_b, y_global_input_a, y_global_input_b = self.get_multi_batch("test")
        return x_input_a, x_input_b, y_input_a, y_input_b, y_global_input_a, y_global_input_b

    def get_val_batch(self):
        """
        Provides a val batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for evaluation
        """
        x_input_a, x_input_b, y_input_a, y_input_b, y_global_input_a, y_global_input_b = self.get_multi_batch("val")
        return x_input_a, x_input_b, y_input_a, y_input_b, y_global_input_a, y_global_input_b

    def get_gen_batch(self):
        """
        Provides a gen batch
        :return: Returns a single data batch (i.e. x_i) to be used for generation on unseen data
        """
        x_input_a = self.get_multi_batch("gen")
        return x_input_a


class DAGANImbalancedDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training):
        """
                :param batch_size: The batch size to use for the data loader
                :param last_training_class_index: The final index for the training set, used to restrict the training set
                if needed. E.g. if training set is 1200 classes and last_training_class_index=900 then only the first 900
                classes will be used
                :param reverse_channels: A boolean indicating whether we need to reverse the colour channels e.g. RGB to BGR
                :param num_of_gpus: Number of gpus to use for training
                :param gen_batches: How many batches to use from the validation set for the end of epoch generations
                """
        self.x_train, self.x_test, self.x_val = self.load_dataset(last_training_class_index)
        print('data shape', self.x_train.shape())

        self.training_data_size = np.sum([len(self.x_train[i]) for i in range(self.x_train.shape[0])])
        self.validation_data_size = np.sum([len(self.x_val[i]) for i in range(self.x_val.shape[0])])
        self.testing_data_size = np.sum([len(self.x_test[i]) for i in range(self.x_test.shape[0])])
        self.generation_data_size = gen_batches * batch_size

        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.reverse_channels = reverse_channels
        self.support_number = support_number

        val_dict = dict()
        idx = 0
        for i in range(self.x_val.shape[0]):
            temp = self.x_val[i]
            for j in range(len(temp)):
                val_dict[idx] = {"sample_idx": j, "label_idx": i}
                idx += 1
        choose_gen_samples = np.random.choice([i for i in range(self.validation_data_size)],
                                              size=self.generation_data_size)

        self.x_gen = np.array([self.x_val[val_dict[idx]["label_idx"]][val_dict[idx]["sample_idx"]]
                               for idx in choose_gen_samples])

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.indexes = {"train": 0, "val": 0, "test": 0, "gen": 0}
        self.datasets = {"train": self.x_train, "gen": self.x_gen,
                         "val": self.x_val,
                         "test": self.x_test}

        self.gen_data_size = gen_batches * self.batch_size
        self.image_height = self.x_train[0][0].shape[0]
        self.image_width = self.x_train[0][0].shape[1]
        self.image_channel = self.x_train[0][0].shape[2]

    def get_batch(self, set_name):
        """
        Generates a data batch to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc
        :return: A data batch
        """
        choose_classes = np.random.choice(len(self.datasets[set_name]), size=self.batch_size)

        x_input_batch_a = []
        x_input_batch_b = []

        for i in range(self.batch_size):
            choose_samples = np.random.choice(len(self.datasets[set_name][choose_classes[i]]),
                                              size=self.support_number * self.batch_size,
                                              replace=False)

            choose_samples_a = choose_samples[:self.batch_size]
            choose_samples_b = choose_samples[self.batch_size:]
            current_class_samples = self.datasets[set_name][choose_classes[i]]

            x_input_batch_a.append(current_class_samples[choose_samples_a[i]])
            x_input_batch_b.append(current_class_samples[choose_samples_b[i]])

        x_input_batch_a = np.array(x_input_batch_a)
        x_input_batch_b = np.array(x_input_batch_b)

        return self.preprocess_data(x_input_batch_a), self.preprocess_data(x_input_batch_b)

    def get_next_gen_batch(self):
        """
        Provides a batch that contains data to be used for generation
        :return: A data batch to use for generation
        """
        if self.indexes["gen"] >= self.gen_data_size:
            self.indexes["gen"] = 0
        x_input_batch_a = self.datasets["gen"][self.indexes["gen"]:self.indexes["gen"] + self.batch_size]
        self.indexes["gen"] += self.batch_size
        return self.preprocess_data(x_input_batch_a)

    def get_multi_batch(self, set_name):
        """
        Returns a batch to be used for training or evaluation for multi gpu training
        :param set_name: The name of the data-set to use e.g. "train", "test" etc
        :return: Two batches (i.e. x_i and x_j) of size [num_gpus, batch_size, im_height, im_width, im_channels). If
        the set is "gen" then we only return a single batch (i.e. x_i)
        """
        x_input_a_batch = []
        x_input_b_batch = []
        if set_name == "gen":
            x_input_a = self.get_next_gen_batch()
            for n_batch in range(self.num_of_gpus):
                x_input_a_batch.append(x_input_a)
            x_input_a_batch = np.array(x_input_a_batch)
            return x_input_a_batch
        else:
            for n_batch in range(self.num_of_gpus):
                x_input_a, x_input_b = self.get_batch(set_name)
                x_input_a_batch.append(x_input_a)
                x_input_b_batch.append(x_input_b)

            x_input_a_batch = np.array(x_input_a_batch)
            x_input_b_batch = np.array(x_input_b_batch)

            return x_input_a_batch, x_input_b_batch


#### 1200:212:211
class OmniglotDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(OmniglotDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                   gen_batches, support_number, is_training,
                                                   general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        ##### generation images for the unseen categories for visualization
        # self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/test_omniglot_c31_s28_data.npy")
        # self.x = self.x / 255
        # x_train, x_val, x_test = self.x[:12], self.x[0:12], self.x[:]

        self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/omniglot_data.npy")
        self.x = self.x / np.max(self.x)
        x_train, x_val, x_test = self.x[:1200], self.x[1200:1412], self.x[1412:]
        # x_train = x_train[:gan_training_index]
        return x_train, x_test, x_val


class OmniglotImbalancedDAGANDataset(DAGANImbalancedDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        super(OmniglotImbalancedDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels,
                                                             num_of_gpus, gen_batches, support_number)

    def load_dataset(self, last_training_class_index):
        x = np.load("../Matching-DAGAN-1wayKshot/datasets/omniglot_data.npy")
        # x = np.load("../Matching-DAGAN-1wayKshot/datasets/test_omniglot_c31_s28_data.npy")
        x_temp = []
        for i in range(x.shape[0]):
            choose_samples = np.random.choice([i for i in range(1, 15)])
            x_temp.append(x[i, :choose_samples])
        self.x = np.array(x_temp)
        self.x = self.x / np.max(self.x)
        x_train, x_val, x_test = self.x[:1200], self.x[1200:1412], self.x[1412:]
        # x_train, x_val, x_test = self.x[:12], self.x[0:12], self.x[:]

        # x_train = x_train[:last_training_class_index]

        return x_train, x_test, x_val


### 1803:500:322 64*64*3
class VGGFaceDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(VGGFaceDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                  gen_batches, support_number, is_training,
                                                  general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        self.x = np.load("./Augmentednpy/dataset_forquality/")
        self.x = self.x / 255
        x_train, x_val, x_test = self.x[:], self.x[:], self.x[:]
        print('data shape', np.shape(self.x))
        return x_train, x_test, x_val


# (149, 100, 84, 84, 3)
class animalsDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(animalsDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                  gen_batches, support_number, is_training,
                                                  general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        # self.x = np.load("./Augmentednpy/dataset/animals_35.npy")
        self.x = np.load("./Augmentednpy/dataset_forquality/animals_100.npy")
        print('data shape', np.shape(self.x))
        self.x = self.x / 255
        x_train, x_val, x_test = self.x[:], self.x[:], self.x[:]
        return x_train, x_test, x_val


## data shape (102, 40, 84, 84, 3)
class flowersDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(flowersDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                  gen_batches, support_number, is_training,
                                                  general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        # self.x = np.load("./Augmentednpy/dataset/flowers_35.npy")
        self.x = np.load("./Augmentednpy/dataset_forquality/flowers_100.npy")
        print('data shape', np.shape(self.x))
        self.x = self.x / 255
        x_train, x_val, x_test = self.x[:], self.x[:], self.x[:]
        return x_train, x_test, x_val


### 10000:5000:1000 28*28*1
class FIGRDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(FIGRDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                               gen_batches, support_number, is_training, general_classification_samples,
                                               selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/FIGR_1_8_data.npy")
        self.x = self.x / np.max(self.x)
        x_train, x_val, x_test = self.x[:10000], self.x[10000:15000], self.x[15000:]
        # x_train = x_train[:gan_training_index]
        return x_train, x_test, x_val


class mnistDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(mnistDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                gen_batches, support_number, is_training,
                                                general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/mnist.npy")
        self.x = self.x / np.max(self.x)
        x_train, x_val, x_test = self.x[:2], self.x[2:9], self.x[9:]
        # x_train = x_train[:gan_training_index]
        return x_train, x_test, x_val


#### 35:7:5 28*28*1
class emnistDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(emnistDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                 gen_batches, support_number, is_training,
                                                 general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/emnist.npy")
        self.x = self.x / np.max(self.x)
        x_train, x_val, x_test = self.x[:35], self.x[35:42], self.x[42:]

        # self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/test_emnist_c38_s28_data.npy")
        # self.x = self.x / 255
        # x_train, x_val, x_test = self.x[:], self.x[:], self.x[:]

        # x_train = x_train[:gan_training_index]
        return x_train, x_test, x_val


### 60:20:20 84*84*3
class miniImagenetDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(miniImagenetDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels,
                                                       num_of_gpus,
                                                       gen_batches, support_number, is_training,
                                                       general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        x_train = np.load("../Matching-DAGAN-1wayKshot/datasets/mini_imagenet_train_3_600_data.npy")
        self.x = x_train
        print('data shape', np.shape(x_train))
        # print('here',np.min(x_train[:100],axis=(0,1,2,3)),np.mean(x_train[:100],axis=(0,1,2,3)),np.max(x_train[:100],axis=(0,1,2,3)),np.std(x_train[:100],axis=(0,1,2,3)))

        x_train = x_train / np.max(x_train)
        # x_train = x_train[:gan_training_index]

        x_test = np.load("../Matching-DAGAN-1wayKshot/datasets/mini_imagenet_test_3_600_data.npy")
        x_test = x_test / np.max(x_test)

        x_val = np.load("../Matching-DAGAN-1wayKshot/datasets/mini_imagenet_val_3_600_data.npy")
        x_val = x_val / np.max(x_val)
        return x_train, x_test, x_val


class FC100DAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(FC100DAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                gen_batches, support_number, is_training,
                                                general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        x_train = np.load("../Matching-DAGAN-1wayKshot/datasets/FC100_train_3_600_3_600_data.npy")
        self.x = x_train
        print('data shape', np.shape(x_train))
        # print('here',np.min(x_train[:100],axis=(0,1,2,3)),np.mean(x_train[:100],axis=(0,1,2,3)),np.max(x_train[:100],axis=(0,1,2,3)),np.std(x_train[:100],axis=(0,1,2,3)))

        x_train = x_train / np.max(x_train)
        # x_train = x_train[:gan_training_index]

        x_test = np.load("../Matching-DAGAN-1wayKshot/datasets/FC100_test_3_600_3_600_data.npy")
        x_test = x_test / np.max(x_test)

        x_val = np.load("../Matching-DAGAN-1wayKshot/datasets/FC100_val_3_600_3_600_data.npy")
        x_val = x_val / np.max(x_val)
        return x_train, x_test, x_val


# (82, 30, 84, 84, 3)
class flowersselectedDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(flowersselectedDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels,
                                                          num_of_gpus,
                                                          gen_batches, support_number, is_training,
                                                          general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/flowers_3_30_selected_3_30_data.npy")
        print('data shape', np.shape(self.x))
        self.x = self.x / np.max(self.x)
        x_train, x_val, x_test = self.x[:70], self.x[30:70], self.x[70:]
        return x_train, x_test, x_val


# (200, 40, 84, 84, 3)
class birdsDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 support_number, is_training, general_classification_samples, selected_classes, image_size):
        super(birdsDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                gen_batches, support_number, is_training,
                                                general_classification_samples, selected_classes, image_size)

    def load_dataset(self, gan_training_index):
        self.x = np.load("../Matching-DAGAN-1wayKshot/datasets/cub_3_40_selected_3_40_data.npy")
        print('data shape', np.shape(self.x))
        # print('here',np.min(self.x[:100],axis=(0,1,2,3)),np.mean(self.x[:100],axis=(0,1,2,3)),np.max(self.x[:100],axis=(0,1,2,3)),np.std(self.x[:100],axis=(0,1,2,3)))
        self.x = self.x / np.max(self.x)

        # self.x = np.reshape(self.x, newshape=(2354, 100, 64, 64, 3))
        x_train, x_val, x_test = self.x[:100], self.x[100:150], self.x[150:]
        # x_train = x_train[:gan_training_index]

        return x_train, x_test, x_val

# data = flowersDAGANDataset(batch_size=1, last_training_class_index=900, reverse_channels=True,
#                                     num_of_gpus=1, gen_batches=1000, support_number=1,is_training=True, general_classification_samples=5,selected_classes=5)


# x_input_batch_a, x_input_batch_b, y_input_batch_a, y_input_batch_b, y_global_input_batch_a, y_global_input_batch_b = data.get_batch('train')
# print(np.max(x_input_batch_a))
# print(np.min(x_input_batch_a))
# print(np.max(y_input_batch_a,axis=1))
# print(np.shape(y_global_input_batch_a))
# # print(np.max(y_global_input_batch_a,axis=1))
# print(x_input_batch_a[0][0][:3])
