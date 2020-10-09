import argparse
import data_with_matchingclassifier as dataset
from generation_general_fewshot_builder_with_matchingclassifier import ExperimentBuilder
import numpy as np

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='batch_size for experiment')
parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=3, help='discr_number_of_conv_per_layer')
parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=3, help='discr_number_of_conv_per_layer')
parser.add_argument('--experiment_title', nargs="?", type=str, default="densenet_generator_fc", help='Experiment name')
parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=1, help='continue from checkpoint of epoch')
parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
parser.add_argument('--dropout_rate_value', type=float, default=0.5, help='dropout_rate_value')
parser.add_argument('--use_wide_connections', nargs="?", type=str, default="False",
                    help='Whether to use wide connections in discriminator')

parser.add_argument('--image_width', nargs="?", type=int, default=128)
parser.add_argument('--image_height', nargs="?", type=int, default=128)
parser.add_argument('--image_channel', nargs="?", type=int, default=3)
parser.add_argument('--matching', nargs="?", type=int, default=1)
parser.add_argument('--fce', nargs="?", type=int, default=0)
parser.add_argument('--full_context_unroll_k', nargs="?", type=int, default=4)
parser.add_argument('--average_per_class_embeddings', nargs="?", type=int, default=0)
parser.add_argument('--dataset',type=str, default='omniglot')

parser.add_argument('--loss_G',type=float, default=1)
parser.add_argument('--loss_D', type=float, default=1)
parser.add_argument('--loss_KL',type=float, default=0.0001)
parser.add_argument('--loss_CLA',type=float, default=1)
parser.add_argument('--loss_FSL',type=float, default=1)
parser.add_argument('--loss_recons_B',type=float, default=0.01)
parser.add_argument('--loss_matching_G',type=float, default=0.01)
parser.add_argument('--loss_matching_D',type=float, default=0.01)
parser.add_argument('--loss_sim',type=float, default=1e2)
parser.add_argument('--strategy', nargs="?", type=int, default=2)


    



###### generating data
parser.add_argument('--support_number', nargs="?", type=int, default=5, help='num_support')
parser.add_argument('--selected_classes',type=int, default=5)



##### general classifier parameters
parser.add_argument('--classification_total_epoch',type=int, default=20)
parser.add_argument('--general_classification_samples',type=int, default=5) ####=support number



##### fewshot classifier parameters
parser.add_argument('--episodes_number',type=int, default=10)
parser.add_argument('--few_shot_episode_classes',type=int, default=1) ###=selected_images





##### matchingGAN related parameters
parser.add_argument('--is_z2', nargs="?", type=int, default=0)
parser.add_argument('--is_z2_vae', nargs="?", type=int, default=0)
parser.add_argument('--restore_path', nargs="?", type=str, default="omniglot_dagan_experiment",
                    help='Experiment name')
parser.add_argument('--augmented_number',type=int, default=0)
parser.add_argument('--num_generations', nargs="?", type=int, default=0, help='num_generations')
parser.add_argument('--confidence',type=int, default=1)
parser.add_argument('--loss_d',type=int, default=1)


###### classifier related parameters
parser.add_argument('--pretrained_epoch',type=int, default=0)
parser.add_argument('--restore_classifier_path', nargs="?", type=str, default="omniglot_dagan_experiment",
                    help='Experiment name')

parser.add_argument('--is_training', nargs="?", type=int, default=0)
parser.add_argument('--is_fewshot_setting', nargs="?", type=int, default=0)



args = parser.parse_args()
args_dict = vars(args)
for key in list(args_dict.keys()):
    print(key, args_dict[key])

batch_size = args.batch_size
num_gpus = args.num_of_gpus


##### generating the batches data
support_num = args.general_classification_samples
selected_classes_num = args.few_shot_episode_classes



if args.dataset == 'omniglot':
    print('omniglot')
    data = dataset.OmniglotDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes, image_size=args.image_width)

elif args.dataset == 'vggface':
    print('vggface')
    data = dataset.VGGFaceDAGANDataset(batch_size=batch_size, last_training_class_index=1600, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'miniimagenet':
    print('miniimagenet')
    data = dataset.miniImagenetDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'emnist':
    print('emnist')
    data = dataset.emnistDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'figr':
    print('figr')
    data = dataset.FIGRDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)
elif args.dataset == 'fc100':
    data = dataset.FC100DAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)
elif args.dataset == 'animals':
    data = dataset.animalsDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'flowers':
    data = dataset.flowersDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'flowersselected':
    data = dataset.flowersselectedDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)


elif args.dataset == 'birds':
    data = dataset.birdsDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)




print('curret test dataset is:',args.dataset)
print('current testing data size is:',np.shape(data.x_test))


experiment = ExperimentBuilder(parser, data=data)
experiment.run_experiment()
