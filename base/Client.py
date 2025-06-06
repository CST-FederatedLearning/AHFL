import copy
import datetime as dt
import os
import pickle as pkl

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn, optim
import torch.nn.functional as F

import copy


class Client(object):
    def __init__(self, name, model, users_dis_model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local=None, test_dl_local=None, mu=0.001, num_classes=10,
                 num_clients=20, client_pred=True, distribution=None):

        self.name = name
        self.net = model
        self.dis_net = users_dis_model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0
        self.count = 0
        self.save_best = True
        self.mu = mu
        self.num_clients = num_clients
        self.client_pred = client_pred
        self.distribution = distribution

    def train(self, client_idx, is_print=False, args=None, lr=0.1):
        self.net.to(self.device)
        self.dis_net.to(self.device)

        self.net.train()
        self.dis_net.train()
        self.client_idx = client_idx
        print(f'lr:{lr}')
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=self.momentum, weight_decay=0)

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)

                self.net.zero_grad()
                self.dis_net.zero_grad()
                log_probs = self.net(images)
                dis_probs = self.dis_net(images)
                dis_pred = self.softmax(dis_probs)
                dis_pred_index = torch.Tensor.cpu(dis_pred).detach().numpy().argmax(axis=1)
                # todo :
                # 增加客户端预测
                loss = 0
                client_idx_loss = 0.0
                if self.client_pred:
                    _out = 0
                    if self.distribution is not None:
                        client_dis = self.distribution[dis_pred_index]
                        _out = client_dis * log_probs
                    if not torch.is_tensor(client_idx):
                        client_idx = torch.Tensor(np.repeat(client_idx, dis_probs.size(0))).type(torch.LongTensor)
                    else:
                        if client_idx.size(0) != dis_probs.size(0):
                            client_idx = torch.Tensor(np.repeat(self.client_idx, dis_probs.size(0))).type(
                                torch.LongTensor)
                            client_idx = client_idx[0:dis_probs.size(0)]
                    client_idx = client_idx.cuda()
                    client_idx_loss += self.loss_func(dis_probs, client_idx)
                    loss += args.ahfl_a * client_idx_loss
                    loss += args.ahfl_b * self.loss_func(_out, labels)

                l = self.loss_func(log_probs, labels)
                loss += args.ahfl_c * self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(l.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def test(self, is_print=False):
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        global_weight_collector = list(self.net.parameters())

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)

                self.net.zero_grad()
                # optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)

                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.net.parameters()):
                    fed_prox_reg += (
                            (self.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss += fed_prox_reg

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self):
        return self.net.state_dict()

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= (len(self.ldr_test.dataset) + 1e-6)
        accuracy = 100. * correct / (len(self.ldr_test.dataset) + 1e-6)
        return test_loss, accuracy

    def eval_test_ahfl(self):
        self.net.to(self.device)
        self.dis_net.to(self.device)
        self.net.eval()
        self.dis_net.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                dis_output = self.dis_net(data)
                dis_pred = dis_output.data.max(1, keepdim=True)[1]
                client_dis = self.distribution[dis_pred]
                output = self.net(data)
                output = client_dis * output
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy

    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy


def dist_layer(_input_planes, output_plans):
    client_layer = nn.Sequential(nn.Linear(_input_planes, 100), nn.LeakyReLU(), nn.Dropout(0.1),
                                 nn.Linear(100, 100), nn.LeakyReLU(), nn.Dropout(0.1),
                                 nn.Linear(100, 200), nn.LeakyReLU(), nn.Dropout(0.5),
                                 nn.Linear(200, 200), nn.LeakyReLU(), nn.Dropout(0.5),
                                 nn.Linear(200, 100), nn.LeakyReLU(), nn.Dropout(0.5),
                                 nn.Linear(100, 100), nn.LeakyReLU(), nn.Dropout(0.1),
                                 nn.Linear(100, output_plans))
    return client_layer


class DisLayer(nn.Module):
    def __init__(self, input_planes, output_planes):
        super().__init__()
        self.input_planes = input_planes
        self.output_planes = output_planes
        # self.client_layer = ResNet9(self.input_planes, self.output_planes)
        self.client_layer = dist_layer(self.input_planes, self.output_planes)
        self.shortcut = nn.Sequential(
            nn.Linear(output_planes, 10), nn.ReLU(), nn.Dropout(0.1),
            # nn.Linear(10, 10), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(10, output_planes)
        )

    def forward(self, X):
        out = self.client_layer(X)
        # out += self.shortcut(X)
        return out


def client(args):
    train_set, user_groups, criterion, args, batch_size, train_params, round_idx, global_model, \
    local_model, client_train_loader, level, scale, h_scale_ratio, client_idx = args

    if args.use_valid:
        local_model = local_model.cuda()

    base_params = [v for k, v in local_model.named_parameters() if 'ee_' not in k]
    exit_params = [v for k, v in local_model.named_parameters() if 'ee_' in k]

    optimizer = torch.optim.SGD([{'params': base_params},
                                 {'params': exit_params}],
                                lr=train_params['lr'],
                                momentum=train_params['momentum'],
                                weight_decay=train_params['weight_decay'])

    loss = 0.0
    for epoch in range(train_params['num_epoch']):
        print(f'{client_idx}-{epoch}-{dt.datetime.now()}')
        iter_idx = round_idx
        loss = execute_epoch(local_model, client_train_loader, criterion, optimizer, iter_idx, epoch,
                             args, train_params, h_scale_ratio, level, global_model)


    print(f'Finished epochs for {client_idx}')
    local_weights = {k: v.cpu() for k, v in local_model.state_dict(keep_vars=True).items()}
    local_grad_flags = {k: v.grad is not None for k, v in local_model.state_dict(keep_vars=True).items()}

    del local_model
    torch.cuda.empty_cache()

    return local_weights, local_grad_flags, loss