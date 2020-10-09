import argparse
import data_with_matchingclassifier as origin_dataset

import data_for_augmentedimages_for_quality_evaluation as test_dataset
# from generation_builder import ExperimentBuilder
from GAN_Metrics_Tensorflow.frechet_kernel_Inception_distance import *
from GAN_Metrics_Tensorflow.inception_score import *
# from GAN_Metrics_Tensorflow.calculate_FID import *
from  GAN_Metrics_Tensorflow.calculate_FID import * 
from  GAN_Metrics_Tensorflow.calculate_FID_tensorflow import * 
from  GAN_Metrics_Tensorflow.calculate_FID_starganv2 import * 

import numpy as np
from glob import glob
import os
import cv2

import torch
# from torchvision import datasets, transforms
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torchvision.datasets as datasets
import torch
import torchvision.transforms as transforms
import cv2
from scipy.io import loadmat
import os
import shutil

def find_classes(root_dir):
    retour = []
    # print('1', root_dir)
    for (root, dirs, files) in os.walk(root_dir):
        # print('origin file',files)
        files.sort()
        # print('origin file',files)
        for f in files:
            if (f.endswith("png")):
                # if (f.endswith("jpg")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour

def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx



class FIGR_Omniglot(data.Dataset):
    def __init__(self, root, dataset, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        if self.dataset == 'Omniglot':
            self.all_items = find_classes(os.path.join(self.root, 'processed'))
        elif self.dataset == 'FIGR' or self.dataset == 'small_FIGR':
            self.all_items = find_classes(self.root)
        else:
            self.all_items = find_classes(self.root)
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])


        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.all_items)

        
def one_channel_evaluation(data_dir, dataset, length, channels=1):
    dset = FIGR_Omniglot(data_dir, dataset, transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                                          lambda x: x.resize((length, length)),
                                                                          lambda x: np.reshape(x, (
                                                                          length, length, channels)),

                                                                          ]))
    return dset


def three_channel_evaluation(data_dir, dataset, length, channels=1):
    dset = FIGR_Omniglot(data_dir, dataset, transform=transforms.Compose([lambda x: cv2.imread(x),
                                                                          lambda x: cv2.resize(x, (length, length),
                                                                                               interpolation=cv2.INTER_LINEAR)

                                                                          ]))                                                              
    return dset


###### forming dataframe from the images files to adapt to our few-shot setting
def generate_image_label_pairs(dataroot, dataset, image_size, channels, each_class_total_samples):
  if channels == 1:
      dataloader = one_channel_evaluation(dataroot, dataset, image_size)
  else:
      dataloader = three_channel_evaluation(dataroot, dataset, image_size)

  classes = len(dataloader)
 
  temp = dict()
  for (img, label) in dataloader:
      if label in temp:
          temp[label].append(img)
      else:
          temp[label] = [img]
  dataloader = []  # Free memory
  for classes in temp.keys():
      # print('here',temp[list(temp.keys())[classes]])
      dataloader.append(np.array(temp[list(temp.keys())[classes]]))
  dataloader = np.array(dataloader)
  temp = []  # Free memory

  shuffle_classes = np.arange(dataloader.shape[0])
  # np.random.shuffle(shuffle_classes)
  dataloader = np.array(
      [dataloader[i][:each_class_total_samples, :, :, :] for i in shuffle_classes if
       np.shape(dataloader[i])[0] >= each_class_total_samples])
  # print('data shape data_loader', np.shape(dataloader))
  return dataloader


  # samples_index = np.random.choice(self.datasets[dataset_name].shape[1], size=samples_number_each_category, replace=True)
  # total_samples = np.zeros([shuffle_classes, each_class_total_samples, image_size, image_size, channels])
  # for i in range(shuffle_classes):
  #     for j in range(each_class_total_samples):
  #         total_samples[i][j] = dataloader[i][j]
  # total_samples = total_samples * 255
  # print('data shape', np.shape(total_samples))
  # return total_samples




# class generate_image_label_pairs():
#     def __init__(self, dataroot, store_path, dataset, image_size, channels=1, each_class_total_samples=8):

#         self.image_size = image_size
#         self.channels = channels
#         self.each_class_total_samples = each_class_total_samples
#         self.dataroot = dataroot
#         self.dataset = dataset
#         self.npy_file = store_path
#         if not os.path.exists(self.npy_file):
#             os.makedirs(self.npy_file)
#     def __call__(self, dataroot, store_path, dataset, image_size, channels=1, each_class_total_samples=8):
#         if self.channels == 1:
#             self.dataloader = one_channel_evaluation(dataroot, self.dataset, self.image_size)
#         else:
#             self.dataloader = three_channel_evaluation(dataroot, self.dataset, self.image_size)

#         classes = len(self.dataloader)
       
#         temp = dict()
#         for (img, label) in self.dataloader:
#             if label in temp:
#                 temp[label].append(img)
#             else:
#                 temp[label] = [img]
#         self.dataloader = []  # Free memory
#         for classes in temp.keys():
#             # print('here',temp[list(temp.keys())[classes]])
#             self.dataloader.append(np.array(temp[list(temp.keys())[classes]]))
#         self.dataloader = np.array(self.dataloader)
#         temp = []  # Free memory

#         shuffle_classes = np.arange(self.dataloader.shape[0])
#         # np.random.shuffle(shuffle_classes)
#         self.dataloader = np.array(
#             [self.dataloader[i][:self.each_class_total_samples, :, :, :] for i in shuffle_classes if
#              np.shape(self.dataloader[i])[0] >= self.each_class_total_samples])
#         # print('data shape', np.shape(self.dataloader))

  
#         # samples_index = np.random.choice(self.datasets[dataset_name].shape[1], size=samples_number_each_category, replace=True)
#         total_samples = np.zeros([shuffle_classes, self.each_class_total_samples, self.image_size, self.image_size, self.channels])
#         for i in range(shuffle_classes):
#             for j in range(self.each_class_total_samples):
#                 total_samples[i][j] = self.dataloader[i][j]
#         total_samples = total_samples * 255
#         print('data shape', np.shape(total_samples))
#         return total_samples





def resize_image(image):
    image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_LINEAR)
    return image


def get_real_fake_images(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples):
    real_images = generate_image_label_pairs(dataroot_real,dataset,image_size, channels, each_class_total_samples)
    

    # real_images = origin_data.get_total_batch_images('test', args.samples_each_category)
    print('real images shape', np.shape(real_images))
    real_images_after = np.zeros([np.shape(real_images)[0],np.shape(real_images)[1], 299, 299, np.shape(real_images)[4]])

    ### real and fake images [category, samples, width, height, channel]
    for i in range(np.shape(real_images)[0]):
      for j in range(np.shape(real_images)[1]):
        resized_image = resize_image(real_images[i][j])
        if len(np.shape(resized_image)) < 3:
            resized_image = np.expand_dims(resized_image, axis=-1)
        real_images_after[i][j] = resized_image
    real_images_after = 255 * (real_images_after / np.max(real_images_after))
    real_images_after = np.transpose(real_images_after, axes=[0, 1, 4, 2, 3])
    

    # fake_images = test_data.get_total_batch_images('test', args.samples_each_category)
    fake_images = generate_image_label_pairs(dataroot_fake, dataset,image_size, channels, each_class_total_samples)
    print('fake images shape', np.shape(fake_images))
    fake_images_after = np.zeros([np.shape(fake_images)[0],np.shape(fake_images)[1], 299, 299, np.shape(fake_images)[4]])
    for i in range(np.shape(fake_images)[0]):
      for j in range(np.shape(fake_images)[1]):
        resized_image = resize_image(fake_images[i][j])
        if len(np.shape(resized_image)) < 3:
            resized_image = np.expand_dims(resized_image, axis=-1)
        fake_images_after[i][j] = resized_image
    fake_images_after = 255 * (fake_images_after / np.max(fake_images_after))
    fake_images_after = np.transpose(fake_images_after, axes=[0, 1, 4, 2, 3])

    ##### extending to three channel images for evaluation metrics
    if np.shape(real_images_after)[2] < 3:
        three_channel_real_images = np.concatenate([real_images_after, real_images_after, real_images_after], axis=1)
        three_channel_fake_images = np.concatenate([fake_images_after, fake_images_after, fake_images_after], axis=1)
    else:
        three_channel_real_images = real_images_after
        three_channel_fake_images = fake_images_after

    # print('fake', np.max(three_channel_fake_images), np.min(three_channel_fake_images))
    # print('real', np.max(three_channel_real_images), np.min(three_channel_real_images))

    # print('real images', np.shape(three_channel_real_images))
    # print('fake images', np.shape(three_channel_fake_images))
    return three_channel_real_images, three_channel_fake_images
    # return three_channel_fake_images, three_channel_fake_images


def inception_score(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples):
  ####only for generated images
    _,images = get_real_fake_images(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples)
    BATCH_SIZE = 32
    images = np.reshape(images,[np.shape(images)[0]* np.shape(images)[1], np.shape(images)[2], np.shape(images)[3], np.shape(images)[4]])
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    logits = inception_logits(inception_images)
    IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)
    print("IS: ", IS)
    return IS



def mean_fid(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples):
  real_images, fake_images  = get_real_fake_images(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples)
  real_images_total = np.reshape(real_images, [np.shape(real_images)[0]*np.shape(real_images)[1], np.shape(real_images)[2],np.shape(real_images)[3],np.shape(real_images)[4]])
  fake_images_total =  np.reshape(fake_images, [np.shape(real_images)[0]*np.shape(real_images)[1], np.shape(real_images)[2],np.shape(real_images)[3],np.shape(real_images)[4]])

  fid_value_total = calculate_fid_given_paths(real_images_total, fake_images_total)
  # fid_value_total = calculate_fid_given_paths_tensorflow(paths = [real_images_total, fake_images_total])
  print('total fid is', fid_value_total)
  mFID = 0
  i = 0
  for i in range(np.shape(real_images)[0]):
    # if i > 5:
    #   break
    FID = calculate_fid_given_paths_tensorflow(paths = [real_images[i], fake_images[i]])
    print('{}_category_fid'.format(i), FID)
    mFID+= FID
    i = i + 1

  mFID = mFID / i
  print("mean FID : ", mFID)
  return fid_value_total, mFID



def mean_fid_tensorflow(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples, is_category):
    # real_images, fake_images  = get_real_fake_images(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples)
    # real_images_total = np.reshape(real_images, [np.shape(real_images)[0]*np.shape(real_images)[1], np.shape(real_images)[2],np.shape(real_images)[3],np.shape(real_images)[4]])
    # fake_images_total =  np.reshape(fake_images, [np.shape(real_images)[0]*np.shape(real_images)[1], np.shape(real_images)[2],np.shape(real_images)[3],np.shape(real_images)[4]])
    # print('maxmium',np.max(real_images_total),np.max(fake_images_total))
    
    # fid_value_total = calculate_fid_given_paths(real_images_total, fake_images_total)
    # fid_value_total = calculate_fid_given_paths_tensorflow(paths = [dataroot_real, dataroot_fake], samples_each_categpry=each_class_total_samples)
    # print('total fid is', fid_value_total)
    mFID = 0
    i = 0

    path_list_real=os.listdir(dataroot_real)
    path_list_real.sort()
    # print('real',path_list_real)

    path_list_fake=os.listdir(dataroot_fake)
    path_list_fake.sort()
    # print('fake',path_list_fake)

    for filename in path_list_real:
      current_real_path = dataroot_real + filename
      current_fake_path = dataroot_fake + filename
      FID = calculate_fid_given_paths_tensorflow(paths = [current_real_path, current_fake_path], samples_each_categpry=each_class_total_samples)
      # FID = calculate_fid_given_paths_tensorflow(paths = [real_images[i], fake_images[i]])
      print('{}_category_fid'.format(i), FID)
      mFID+= FID
      i = i + 1

    mFID = mFID / i
    print("mean FID : ", mFID)


def calculate_fid_starganv2(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples, is_category):
  print('Calculating FID for all images...')
  fid_value_total = calculate_fid_given_paths_starganv2(
              paths=[dataroot_real, dataroot_fake],
              img_size=128,
              batch_size=32)
  print('total fid', fid_value_total)
  mFID = 0

  if is_category:
      print('Calculating FID for each category...')
      path_list_real=os.listdir(dataroot_real)
      path_list_real.sort()

      path_list_fake=os.listdir(dataroot_fake)
      path_list_fake.sort()
      fid_values = 0
      i = 0
      for filename in path_list_real:
        path_real = os.path.join(dataroot_real, filename)
        path_fake = os.path.join(dataroot_fake, filename)
        fid_value = calculate_fid_given_paths_starganv2(
            paths=[path_real, path_fake],
            img_size=256,
            batch_size=32)
        fid_value += fid_value
        print('category_{} fid'.format(i),fid_value)
        i = i + 1

      # calculate the average FID for all tasks
      mFID = fid_value / len(path_list_fake)
      print('mean fid', mFID)
  return fid_value_total, mFID




def frechet_inception_distance(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples, is_category):
    real_images, fake_images = get_real_fake_images(dataroot_real, dataroot_fake, dataset,image_size, channels,each_class_total_samples)
    BATCH_SIZE = 32
    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    activations = inception_activations(inception_images)

    ##### total FID
    real_images_total = np.reshape(real_images, [np.shape(real_images)[0]*np.shape(real_images)[1], np.shape(real_images)[2],np.shape(real_images)[3],np.shape(real_images)[4]])
    fake_images_total =  np.reshape(fake_images, [np.shape(real_images)[0]*np.shape(real_images)[1], np.shape(real_images)[2],np.shape(real_images)[3],np.shape(real_images)[4]])
    total_FID = get_fid(fcd, BATCH_SIZE, real_images_total, fake_images_total, inception_images, real_activation, fake_activation,
                    activations)


    print('total FID:', total_FID)
    mFID = 0
    i = 0
    category_FID = []
    repeat_num = int(np.shape(real_images)[0])
    if is_category:
      for i in range(np.shape(real_images)[0]):
        # if i > 5:
        #   break
        FID = get_fid(fcd, BATCH_SIZE, real_images[i], fake_images[i], inception_images, real_activation, fake_activation,
                      activations)
        category_FID.append(FID)
        print('{}_category_fid'.format(i), FID)
        mFID+= FID
        i = i + 1
      mFID = mFID / i
    else:
      category_FID.append(0)

     
    print("mean FID : ", mFID)
    return total_FID, mFID, category_FID





