import argparse
from cal_gan_metrics_from_files import *
import sys
sys.path.append("./LPIPS_pytorch")
from  LPIPS_pytorch.compute_dists_pair import *

import numpy as np
from glob import glob
import os
import cv2


import argparse

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')

parser.add_argument('--dataroot_real', nargs="?", type=str, default='./coarse-data/visualization_training_images/vggface')
parser.add_argument('--dataroot_fake', nargs="?", type=str, default='./coarse-data/visualization_training_images/vggface')
parser.add_argument('--image_width', nargs="?", type=int, default=128)
parser.add_argument('--image_channel', nargs="?", type=int, default=3)
parser.add_argument('--augmented_support', nargs="?", type=int, default=512)

parser.add_argument('-d','--dir', type=str, default='./imgs/ex_dir_pair')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

parser.add_argument('--is_FID', nargs="?", type=int, default=1)
parser.add_argument('--is_IS', nargs="?", type=int, default=0)
parser.add_argument('--is_LPIPS', nargs="?", type=int, default=1)
parser.add_argument('--is_LPIPS_CATEGORY', nargs="?", type=int, default=1)
parser.add_argument('--is_FID_CATEGORY', nargs="?", type=int, default=0)


# fake_images_path = './vggface1way3shotNEW/test/'
# mean_fid(fake_images_path)
#### calculating the IS from whole dataset
args = parser.parse_args()
dataset = str(args.dataroot_real.split('/')[2]) + '_{}'.format(args.augmented_support)


# print('NEW CALCULATING FID')
# calculate_fid_starganv2(dataroot_real=args.dataroot_real, dataroot_fake=args.dataroot_fake, dataset=dataset,
#                            image_size=args.image_width, channels=args.image_channel,
#                            each_class_total_samples=args.augmented_support)

# mean_fid(dataroot_real=args.dataroot_real, dataroot_fake=args.dataroot_fake, dataset=dataset,
#                            image_size=args.image_width, channels=args.image_channel,
#                            each_class_total_samples=args.augmented_support)


f = open(args.out,'w')

if args.is_FID > 0:
	print('CALCULATING FID')
	# fid_value_total, mFID, category_FID = frechet_inception_distance(dataroot_real=args.dataroot_real, dataroot_fake=args.dataroot_fake, dataset=dataset,
	#                            image_size=args.image_width, channels=args.image_channel,
	#                            each_class_total_samples=args.augmented_support,is_category=args.is_FID_CATEGORY)

	fid_value_total, mFID, category_FID = calculate_fid_starganv2(dataroot_real=args.dataroot_real,
																	 dataroot_fake=args.dataroot_fake, dataset=dataset,
																	 image_size=args.image_width,
																	 channels=args.image_channel,
																	 each_class_total_samples=args.augmented_support,
																	 is_category=args.is_FID_CATEGORY)




	f.writelines('FID TOTAL: %.4f \n'%fid_value_total)
	f.writelines('MEAN FID: %.4f \n'%mFID)
	if args.is_FID_CATEGORY:
		for i in range(len(category_FID)):
			f.writelines('FID CATEGORY_%.1f: %.4f \n'%(i, category_FID[i]))



if args.is_LPIPS > 0:
	print('CALCULATING LPIPS')
	dists, dist_mean = compute_dists_pair(args.use_gpu, args.dir, args.out)
	f.writelines('LPIPS MEAN: %.4f \n'%dist_mean)
	for i in range(len(dists)):
		f.writelines('LPIPS CATEGORY_%.1f: %.4f \n'%(i, dists[i]))

if args.is_IS > 0:
	print('CALCULATING IS')
	IS = inception_score(dataroot_real=args.dataroot_real,dataroot_fake=args.dataroot_fake, dataset=dataset,
	                           image_size=args.image_width, channels=args.image_channel,
	                           each_class_total_samples=args.augmented_support)
	f.writelines('IS MEAN: %.4f \n'%IS[0])
	f.writelines('IS VARAINCE: %.4f \n'%IS[1])



f.close()