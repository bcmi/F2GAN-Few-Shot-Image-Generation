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
        for f in files:
            if (f.endswith("jpg")):
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
        print(img,target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return img, target

    def __len__(self):
        return len(self.all_items)


def one_channel_preparation(data_dir, dataset, length, channels, batch_size):
    train_loader_1 = FIGR_Omniglot(data_dir, dataset,
                                   transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                                 lambda x: x.resize((length, length)),
                                                                 lambda x: np.reshape(x, (channels, length, length)),
                                                                 ]))
    train_loader = torch.utils.data.DataLoader(train_loader_1, batch_size=batch_size, shuffle=True)
    return train_loader


def three_channel_preparation(data_dir, length, batch_size):
    transform = transforms.Compose([
        transforms.Scale(length),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dset = datasets.ImageFolder(data_dir, transform)
    train_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle=False)
    return train_loader


def one_channel_evaluation(data_dir, dataset, length, channels=1):
    dset = FIGR_Omniglot(data_dir, dataset, transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                                          lambda x: x.resize((length, length)),
                                                                          lambda x: np.reshape(x, (
                                                                          length, length, channels)),

                                                                          ]))
    return dset


def three_channel_evaluation(data_dir, dataset, length, channels=1):
    # transform = transforms.Compose([
    #         transforms.Scale(length),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # dset = datasets.ImageFolder(data_dir, transform)
    dset = FIGR_Omniglot(data_dir, dataset, transform=transforms.Compose([lambda x: cv2.imread(x),
                                                                          lambda x: cv2.resize(x, (length, length),
                                                                                               interpolation=cv2.INTER_LINEAR)

                                                                          ]))

    # dset = FIGR_Omniglot(data_dir, dataset, transform=transforms.Compose([transforms.Scale(length),
    #                                                                         transforms.ToTensor(),
    #                                                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

    return dset


###### forming dataframe from the images files to adapt to our few-shot setting
class generate_image_label_pairs():
    def __init__(self, dataroot, store_path, dataset, image_size, channels=1, each_class_total_samples=8):

        self.image_size = image_size
        self.channels = channels
        self.each_class_total_samples = each_class_total_samples
        self.dataroot = dataroot
        self.dataset = dataset
        self.npy_file = store_path
        if not os.path.exists(self.npy_file):
            os.makedirs(self.npy_file)

        if self.channels == 1:
            self.dataloader = one_channel_evaluation(dataroot, self.dataset, self.image_size)
        else:
            self.dataloader = three_channel_evaluation(dataroot, self.dataset, self.image_size)

        classes = len(self.dataloader)
        save_file_data = self.npy_file + '{}.npy'.format(self.dataset)
        print('store_path',save_file_data)

        if not os.path.isfile(save_file_data):
            temp = dict()
            for (img, label) in self.dataloader:
                if label in temp:
                    temp[label].append(img)
                else:
                    temp[label] = [img]
            self.dataloader = []  # Free memory
            for classes in temp.keys():
                # print('here',temp[list(temp.keys())[classes]])
                self.dataloader.append(np.array(temp[list(temp.keys())[classes]]))
            self.dataloader = np.array(self.dataloader)
            temp = []  # Free memory

            shuffle_classes = np.arange(self.dataloader.shape[0])
            np.random.shuffle(shuffle_classes)
            self.dataloader = np.array(
                [self.dataloader[i][:self.each_class_total_samples, :, :, :] for i in shuffle_classes if
                 np.shape(self.dataloader[i])[0] >= self.each_class_total_samples])
            print('data shape', np.shape(self.dataloader))
            np.save(save_file_data, self.dataloader)


####### loading mnist dataset for our setting
import numpy as np
from urllib import request
import gzip
import pickle

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


###### final_data(10, 6300, 28, 28, 1)
def mnist_formation(save_file_data):
    x_train, y_train, x_test, y_test = load()
    x_train = np.reshape(x_train, [np.shape(x_train)[0], 28, 28, 1])
    x_test = np.reshape(x_test, [np.shape(x_test)[0], 28, 28, 1])
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    final_data = []
    if not os.path.isfile(save_file_data):
        temp = dict()
        for i in range(np.shape(x)[0]):
            if y[i] in temp:
                temp[y[i]].append(x[i])
            else:
                temp[y[i]] = [x[i]]

        for classes in temp.keys():
            print('here', np.shape(temp[list(temp.keys())[classes]]))
            final_data.append(np.array(temp[list(temp.keys())[classes]][:6300]))
        final_data = np.array(final_data)
        temp = []
    np.save(save_file_data, final_data)


# save_file_data = './datasets/mnist.npy'
# mnist_formation(save_file_data)


def EMNIST():
    filename = [
        ["training_images", "./datasets/gzip/emnist-balanced-train-images-idx3-ubyte.gz"],
        ["test_images", "./datasets/gzip/emnist-balanced-test-images-idx3-ubyte.gz"],
        ["training_labels", "./datasets/gzip/emnist-balanced-train-labels-idx1-ubyte.gz"],
        ["test_labels", "./datasets/gzip/emnist-balanced-test-labels-idx1-ubyte.gz"]
    ]

    # filename = [
    # ["training_images","emnist-balanced-train-images-idx3-ubyte"],
    # ["test_images","emnist-balanced-test-images-idx3-ubyte"],
    # ["training_labels","emnist-balanced-train-labels-idx1-ubyte"],
    # ["test_labels","emnist-balanced-test-labels-idx1-ubyte"]
    # ]

    emnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            emnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            emnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("emnist.pkl", 'wb') as f:
        pickle.dump(emnist, f)
    print("Save complete.")

    x_train, y_train, x_test, y_test = emnist["training_images"], emnist["training_labels"], emnist["test_images"], \
                                       emnist["test_labels"]
    x_train = np.reshape(x_train, [np.shape(x_train)[0], 28, 28, 1])
    x_test = np.reshape(x_test, [np.shape(x_test)[0], 28, 28, 1])
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    save_file_data = './datasets/emnist.npy'
    final_data = []
    if not os.path.isfile(save_file_data):
        temp = dict()
        for i in range(np.shape(x)[0]):
            if y[i] in temp:
                temp[y[i]].append(x[i])
            else:
                temp[y[i]] = [x[i]]

        for classes in temp.keys():
            final_data.append(np.array(temp[list(temp.keys())[classes]][:6300]))
        final_data = np.array(final_data)
        temp = []
        # print('data',np.shape(final_data)) (47, 2800, 28, 28, 1)
    np.save(save_file_data, final_data)


def flowers(image_dir, label_dir, save_dir):
    label = loadmat(label_dir)
    flower_labels = list(label['labels'][0])

    for index_real, item in enumerate(flower_labels):
        index = index_real + 1
        if os.path.exists(save_dir + '/{}'.format(item)):
            flag = int(index / 10)
            if flag < 1:
                shutil.move(image_dir + 'image_0000{}.jpg'.format(index),
                            save_dir + '/{}/image_0000{}.jpg'.format(item, index))
            elif 1 <= flag < 10:
                shutil.move(image_dir + 'image_000{}.jpg'.format(index),
                            save_dir + '/{}/image_000{}.jpg'.format(item, index))
            elif 10 <= flag < 100:
                shutil.move(image_dir + 'image_00{}.jpg'.format(index),
                            save_dir + '/{}/image_00{}.jpg'.format(item, index))
            elif 100 <= flag < 1000:
                shutil.move(image_dir + 'image_0{}.jpg'.format(index),
                            save_dir + '/{}/image_0{}.jpg'.format(item, index))
        else:
            os.mkdir(save_dir + '/{}'.format(item))
            flag = int(index / 10)
            if flag < 1:
                shutil.move(image_dir + 'image_0000{}.jpg'.format(index),
                            save_dir + '/{}/image_0000{}.jpg'.format(item, index))
            elif 1 <= flag < 10:
                shutil.move(image_dir + 'image_000{}.jpg'.format(index),
                            save_dir + '/{}/image_000{}.jpg'.format(item, index))
            elif 10 <= flag < 100:
                shutil.move(image_dir + 'image_00{}.jpg'.format(index),
                            save_dir + '/{}/image_00{}.jpg'.format(item, index))
            elif 100 <= flag < 1000:
                shutil.move(image_dir + 'image_0{}.jpg'.format(index),
                            save_dir + '/{}/image_0{}.jpg'.format(item, index))


# dataroot = '../GAN_comparison/data/FIGR-8/Data'
# dataset = 'FIGR'
# dataroot = '../GAN_comparison/data/small-FIGR-8'
# dataset = 'small_FIGR'


# dataroot = './datasets/mini_imagenet/train/'
# dataset = 'mini_imagenet_train'
# generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=84, channels=3, each_class_total_samples=600)


# EMNIST()


# dataroot = './coarse-data/animals'
# dataset = 'animals_128'
# generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=128, channels=3, each_class_total_samples=100)


#### for flowers
# image_dir = './datasets/Flowers/flowers/'
# label_dir = './datasets/Flowers/imagelabels.mat'
# save_dir = './datasets/Flowers/flower_folders'
# flowers(image_dir, label_dir, save_dir)

# 102 classes, 8189 items
# dataroot = './coarse-data/Flowers/flower_folders'
# dataset = 'flowers_128'
# generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=128, channels=3, each_class_total_samples=1)


#### for birds
# Found 11788 items
# Found 200 classes
# (200, 40, 84, 84, 3)


####### resize the images 1004
# dataroot = './coarse-data/Flowers/flower_folders'
# dataset = 'flowers'
# generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=128, channels=3, each_class_total_samples=40)


# dataroot = './coarse-data/animals'
# dataset = 'animals'
# generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=128, channels=3, each_class_total_samples=40)


# dataroot = './coarse-data/CUB_200_2011'
# dataset = 'birds'
# generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=128, channels=3, each_class_total_samples=40)


# dataroot = './coarse-data/mini_imagenet'
# dataset = 'mini_imagenet'
# generate_image_label_pairs(dataroot=dataroot, dataset=dataset, image_size=128, channels=3, each_class_total_samples=40)

import argparse

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')

parser.add_argument('--dataroot', nargs="?", type=str, default='./coarse-data/visualization_training_images/vggface')
parser.add_argument('--storepath', nargs="?", type=str,
                    default='./augmented_dataset/visualization_training_images/vggface')
parser.add_argument('--image_width', nargs="?", type=int, default=128)
parser.add_argument('--image_channel', nargs="?", type=int, default=3)
parser.add_argument('--augmented_support', nargs="?", type=int, default=512)

args = parser.parse_args()

dataset = str(args.dataroot.split('/')[2]) + '_{}'.format(args.augmented_support)
generate_image_label_pairs(dataroot=args.dataroot, store_path=args.storepath, dataset=dataset,
                           image_size=args.image_width, channels=args.image_channel,
                           each_class_total_samples=args.augmented_support)



#### python data_preparation.py --dataroot /media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/Matching-DAGAN-1wayKshot/coarse-data/visualization_training_images/emnist --storepath /media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/FusionGAN-SelfAttention-MSIE-IJCAI/TestOmniglotEmnist/ --image_width 28 --image_channel 1 --augmented_support 3













