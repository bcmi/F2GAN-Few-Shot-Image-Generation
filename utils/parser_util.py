import argparse


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_args():
    parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='batch_size for experiment')
    parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1,
                        help='Number of inner layers per multi layer in the discriminator')
    parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=2,
                        help='Number of inner layers per multi layer in the generator')
    parser.add_argument('--experiment_title', nargs="?", type=str, default="omniglot_dagan_experiment",
                        help='Experiment name')

    parser.add_argument('--restore_path', nargs="?", type=str, default="omniglot_dagan_experiment",
                        help='Experiment name')

    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1,
                        help='continue from checkpoint of epoch')
    parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
    parser.add_argument('--dropout_rate_value', type=float, default=0,
                        help='A dropout rate placeholder or a scalar to use throughout the network')
    parser.add_argument('--num_generations', nargs="?", type=int, default=32,
                        help='The number of samples generated for use in the spherical interpolations at the end of '
                             'each epoch')

    parser.add_argument('--support_number', nargs="?", type=int, default=5,
                        help='The number of samples generated for use in the spherical interpolations at the end of '
                             'each epoch')

    parser.add_argument('--use_wide_connections', nargs="?", type=str, default="False",
                        help='Whether to use wide connections in discriminator')

    parser.add_argument('--matching', nargs="?", type=int, default=1)
    parser.add_argument('--fce', nargs="?", type=int, default=0)
    parser.add_argument('--full_context_unroll_k', nargs="?", type=int, default=4)
    parser.add_argument('--average_per_class_embeddings', nargs="?", type=int, default=0)
    parser.add_argument('--is_training', nargs="?", type=int, default=1)

    parser.add_argument('--dataset', type=str, default='omniglot')
    parser.add_argument('--general_classification_samples', type=int, default=5)
    parser.add_argument('--selected_classes', type=int, default=0)

    parser.add_argument('--loss_G', type=float, default=1)
    parser.add_argument('--loss_D', type=float, default=1)
    parser.add_argument('--loss_KL', type=float, default=0.0001)
    parser.add_argument('--loss_CLA', type=float, default=1)
    parser.add_argument('--loss_FSL', type=float, default=1)
    parser.add_argument('--loss_recons_B', type=float, default=0.01)
    parser.add_argument('--loss_matching_G', type=float, default=0.01)
    parser.add_argument('--loss_matching_D', type=float, default=0.01)
    parser.add_argument('--loss_sim', type=float, default=1e2)

    parser.add_argument('--is_z2', nargs="?", type=int, default=0)
    parser.add_argument('--is_z2_vae', nargs="?", type=int, default=0)

    parser.add_argument('--image_width', nargs="?", type=int, default=128)
    parser.add_argument('--image_height', nargs="?", type=int, default=128)
    parser.add_argument('--image_channel', nargs="?", type=int, default=3)
    parser.add_argument('--strategy', nargs="?", type=int, default=2)
    parser.add_argument('--is_all_test_categories', nargs="?", type=int, default=1)
    parser.add_argument('--generation_layers', nargs="?", type=int, default=10)
    parser.add_argument('--is_generation_for_classifier', nargs="?", type=int, default=0)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_gpus = args.num_of_gpus
    support_number = args.support_number

    args_dict = vars(args)
    for key in list(args_dict.keys()):
        print(key, args_dict[key])

        if args_dict[key] == "True":
            args_dict[key] = True
        elif args_dict[key] == "False":
            args_dict[key] = False
    args = Bunch(args_dict)

    return batch_size, num_gpus, support_number, args