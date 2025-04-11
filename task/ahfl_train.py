# ... existing code ...
import time

import numpy as np
import torch
from torch import nn

from utils.distill_utils import accuracy, adjust_learning_rate


class AverageMeter:
    """
    A class used to track and calculate the average value.

    This class records the total value, the number of samples, and the average value,
    and provides functions for updating and resetting these values.
    """
    def __init__(self):
        """
        Initialize an AverageMeter object.

        The initial values are set to 0 for subsequent average value calculations.
        """
        self.reset()

    def reset(self):
        """
        Reset all counters and the average value.

        Set the sum, count, and average to 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the current value, sum, count, and average.

        :param val: The current value.
        :param n: The number of samples for the current value, default is 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def execute_epoch(model, train_loader, criterion, optimizer, round, epoch, args, train_params, h_level, distribution, client_idx=0):
    """
    Execute one training epoch.

    :param model: The model to be trained.
    :param train_loader: The training data loader.
    :param criterion: The loss function.
    :param optimizer: The optimizer.
    :param round: The current round number.
    :param epoch: The current training epoch.
    :param args: The command-line arguments.
    :param train_params: The training parameters.
    :param h_level: The hierarchical level.
    :param distribution: The distribution information.
    :param client_idx: The client index, default is 0.
    :return: The average loss.
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(h_level):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    for i, (inp, target) in enumerate(train_loader):

        adjust_learning_rate(optimizer, round, train_params)

        data_time.update(time.time() - end)

        if args.use_valid:
            inp = inp.cuda()
            target = target.cuda()

        output = model(inp, manual_early_exit_index=h_level)

        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        for j in range(len(output)):
            if j == len(output) - 1:
                loss += criterion.ce_loss(output[j], target) * (j + 1)
            else:
                gamma_active = round > args.num_rounds * 0.25
                loss += criterion.loss_fn_kd(output[j], target, output[-1], gamma_active) * (j + 1)

        for j in range(len(output)):
            if 'bert' in args.arch:
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), inp.size(0))
            top5[j].update(prec5.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss /= len(output) * (len(output) + 1) / 2

        client_idx_loss = 0.0
        # Assume client_pred is a boolean variable used to control whether to execute the client prediction logic
        client_pred = True  # Can be modified according to actual situation
        # Assume dis_pred_index is an integer used to get the client distribution from the distribution
        dis_pred_index = 0  # Can be modified according to actual situation
        # Assume log_probs is a tensor used to calculate the client prediction loss
        log_probs = torch.randn(10)  # Can be modified according to actual situation
        # Assume dis_probs is a tensor used to calculate the client index loss
        dis_probs = torch.randn(10, 10)  # Can be modified according to actual situation
        # Assume labels is a tensor used to calculate the cross-entropy loss
        labels = torch.randint(0, 10, (10,))  # Can be modified according to actual situation
        # Assume distribution is a tensor used to calculate the client distribution loss
        if client_pred:
            _out = 0
            if distribution is not None:
                client_dis = distribution[dis_pred_index]
                _out = client_dis * log_probs
            if not torch.is_tensor(client_idx):
                client_idx = torch.Tensor(np.repeat(client_idx, dis_probs.size(0))).type(torch.LongTensor)
            else:
                if client_idx.size(0) != dis_probs.size(0):
                    client_idx = torch.Tensor(np.repeat(client_idx, dis_probs.size(0))).type(
                        torch.LongTensor)
                    client_idx = client_idx[0:dis_probs.size(0)]
            client_idx = client_idx.cuda()
            client_idx_loss += nn.CrossEntropyLoss()(dis_probs, client_idx)

        loss += args.ahfl_a * client_idx_loss
        loss += args.ahfl_b * nn.CrossEntropyLoss()(_out, labels)
        loss += args.ahfl_c * nn.CrossEntropyLoss()(log_probs, labels)

        losses.update(loss.item(), inp.size(0))

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i + 1}/{len(train_loader)}]\t\t' +
                  f'Exit: {len(output)}\t' +
                  f'Time: {batch_time.avg:.3f}\t' +
                  f'Data: {data_time.avg:.3f}\t' +
                  f'Loss: {losses.val:.4f}\t' +
                  f'Acc@1: {top1[-1].val:.4f}\t' +
                  f'Acc@5: {top5[-1].val:.4f}')

    return losses.avg