import argparse
import datetime
import os


def modify_args(args):
    if args.use_gpu and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.data == 'cifar10':
        args.num_classes = 10
        args.image_size = (32, 32)
    elif args.data == 'cifar100':
        args.num_classes = 100
        args.image_size = (32, 32)
    elif args.data == 'fashionmnist':
        args.num_classes = 10
        args.image_size = (28, 28)
    elif args.data == 'cinic10':
        args.num_classes = 10
        args.image_size = (32, 32)
    else:
        raise NotImplementedError

    if not hasattr(args, "save_path") or args.save_path is None:
        args.save_path = f"./outputs2/{args.arch}_{args.evalmode}_{args.data}_" \
                         f"{format(str(datetime.datetime.now()).replace(':', ' '))}_" \
                         f"{args.num_clients}_r{args.num_rounds}_{args.sample_rate}_a{args.alpha}"

    return args


model_names = ['msdnet24_1', 'msdnet24_4',
               'resnet110_1', 'resnet110_4']

parser = argparse.ArgumentParser(
    description='Image classification PK main script')

exp_group = parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save_path', default=None, type=str, metavar='SAVE', help='Path to the directory where experiment logs will be saved')
exp_group.add_argument('--resume', action='store_true', help='If set, resume training from the latest checkpoint. No specific path needed as it assumes default location.')
exp_group.add_argument('--evalmode', default=None, choices=['local', 'global'], help='Select the evaluation mode: "local" or "global"')
exp_group.add_argument('--evaluate_from', default=None, type=str, metavar='PATH', help='Path to the saved checkpoint file for evaluation. Leave empty if not evaluating from a checkpoint.')
exp_group.add_argument('--print-freq', '-p', default=100, type=int,  metavar='N', help='Frequency (in batches) at which to print training progress. Default is 100.')
exp_group.add_argument('--seed', default=0, type=int, help='Random seed value for reproducibility of experiments.')
exp_group.add_argument('--gpu_idx', default=0, type=str, help='Index of the GPU to use for training. For example, "0" for the first GPU.')
exp_group.add_argument('--use_gpu', default=1, type=int, help='Set to 0 to use CPU for training; otherwise, use GPU.')

# dataset related
data_group = parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10', choices=['cifar10', 'cifar100', 'fashionmnist', 'cinic10'], help='Name of the dataset to use for training and evaluation.')
data_group.add_argument('--data-root', metavar='DIR', default='data', help='Path to the root directory where the dataset is stored. Default is "data".')
data_group.add_argument('--use-valid', default=1, action='store_true', help='If set, use a validation set during training; otherwise, only use training and test sets.')
data_group.add_argument('-j', '--workers', default=5, type=int, metavar='N', help='Number of worker threads to use for data loading. Default is 5.')
data_group.add_argument('-jj', '--num_fed_workers', default=1, type=int, metavar='N', help='Number of workers for federated learning operations. Default is 1.')
# model arch related
arch_group = parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet110_4', type=str, choices=model_names,
                        help='Select the model architecture. Available options are: ' +
                             ' | '.join(model_names) +
                             '. Default is "resnet110_4".')
parser.add_argument('--dis_model', type=str, default='mlp', help='Name of the discriminator model. Default is "mlp".')
arch_group.add_argument('--ee_locs', type=int, nargs='*', default=[], help='List of early exit locations in the model.')

# training related
optim_group = parser.add_argument_group('optimization', 'optimization setting')

optim_group.add_argument('--start_round', default=0, type=int, metavar='N', help='Manually set the starting round number. Useful when resuming training.')
optim_group.add_argument('-b', '--batch-size', type=int, help='Size of each mini-batch during training.')
optim_group.add_argument('--KD_gamma', type=float, default=0, help='Value of the gamma parameter for Knowledge Distillation (KD).')
optim_group.add_argument('--KD_T', type=int, default=3, help='Temperature value (T) for Knowledge Distillation (KD).')

# FL related
fl_group = parser.add_argument_group('fl', 'FL setting')
fl_group.add_argument('--vertical_scale_ratios', type=float, nargs='*', default=[0.7, 0.7, 0.75, 1], help='List of vertical split ratios for the model at each complexity level.')
fl_group.add_argument('--horizontal_scale_ratios', type=int, nargs='*', default=[1, 2, 3, 4], help='List of horizontal split indices for the model at each complexity level.')
fl_group.add_argument('--client_split_ratios', type=float, nargs='*', default=[0.25, 0.25, 0.25, 0.25], help='Ratio of clients assigned to each complexity level in federated learning.')
fl_group.add_argument('--num_rounds', type=int, default=10, help='Total number of training rounds in federated learning.')
fl_group.add_argument('--num_clients', type=int, default=100, help='Total number of clients participating in federated learning.')
fl_group.add_argument('--sample_rate', type=float, default=0.1, help='Sampling rate of clients in each round of federated learning.')
fl_group.add_argument('--alpha', type=int, default=0.1, help='Alpha value representing the degree of non-IID data distribution among clients.')
fl_group.add_argument('-trs', '--track_running_stats', action='store_true', help='If set, track running statistics during training.')

#ahfl_pft
parser.add_argument('--client_pred', type=bool, default=True, help='Enable or disable client prediction.')
parser.add_argument('--ahfl_a', type=float, default=0.4, help='Parameter a for the AHFL-PFT algorithm.')
parser.add_argument('--ahfl_b', type=float, default=0.1, help='Parameter b for the AHFL-PFT algorithm.')
parser.add_argument('--ahfl_c', type=float, default=0.5, help='Parameter c for the AHFL-PFT algorithm.')
parser.add_argument('--lr_i', type=float, default=[50], help='Initial learning rate for a specific component. Note: should be a single float value.')
