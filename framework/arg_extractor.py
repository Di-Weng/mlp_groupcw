import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=1902530,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--input_dim', nargs='?', type=int, default=40, help='Input dimensionality for experiment, equals to n_mfcc')
    parser.add_argument('--hidden_dim', nargs='?', type=int, default=64, help='Hidden dimensionality for experiment')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--num_classes', nargs="?", type=int, default=4, help='The experiment\'s output classes')
    parser.add_argument('--num_layers', nargs="?", type=int, default=1, help='Number of BLSTM layers')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0,
                        help='Weight decay to use for Adam')
    args = parser.parse_args()
    print(args)
    return args
