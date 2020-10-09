import data_with_matchingclassifier as dataset
from experiment_builder_with_matchingclassifier import ExperimentBuilder
from utils.parser_util import get_args

batch_size, num_gpus,support_num, args = get_args()
#set the data provider to use for the experiment


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

    



#init experiment
experiment = ExperimentBuilder(args, data=data)
#run experiment
experiment.run_experiment()

