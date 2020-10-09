import argparse
import data_with_matchingclassifier as dataset
import numpy as np
from classification_builder import ExperimentBuilder
from utils.parser_util import get_args

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='batch_size for experiment')
parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--experiment_title', nargs="?", type=str, default="densenet_generator_fc", help='Experiment name')
parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='continue from checkpoint of epoch')
parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
parser.add_argument('--dropout_rate_value', type=float, default=0.5, help='dropout_rate_value')
parser.add_argument('--num_generations', nargs="?", type=int, default=64, help='num_generations')
parser.add_argument('--support_number', nargs="?", type=int, default=1, help='num_support')
parser.add_argument('--use_wide_connections', nargs="?", type=str, default="False",
                    help='Whether to use wide connections in discriminator')

parser.add_argument('--matching', nargs="?", type=int, default=0)
parser.add_argument('--fce', nargs="?", type=int, default=0)
parser.add_argument('--full_context_unroll_k', nargs="?", type=int, default=4)
parser.add_argument('--average_per_class_embeddings', nargs="?", type=int, default=0)
parser.add_argument('--is_training', nargs="?", type=int, default=0)
parser.add_argument('--classification_total_epoch',type=int, default=200)
parser.add_argument('--dataset',type=str, default='omniglot')
parser.add_argument('--general_classification_samples',type=int, default=5)
parser.add_argument('--selected_classes',type=int, default=1)


parser.add_argument('--image_width',type=int, default=1)
parser.add_argument('--pretrain',type=int, default=0)





args = parser.parse_args()
batch_size = args.batch_size
num_gpus = args.num_of_gpus
support_num = args.support_number

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





experiment = ExperimentBuilder(parser, data=data)
#run experiment
experiment.run_experiment()
